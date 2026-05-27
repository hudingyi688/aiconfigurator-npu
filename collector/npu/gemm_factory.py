"""GEMM operator factory for Ascend NPU benchmarking.

Constructs GEMM operators via vllm-ascend's linear layer abstractions,
ensuring the benchmark exercises the same kernel path as real inference.

Aligned with AIConfigurator collector/vllm/collect_gemm.py:
- 6 independent GEMM instances for L2 cache flush (outside_loop_count=6)
- Returns (forward_func, op_count) so caller divides total latency by op_count
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Quantization type constants
QUANT_BF16 = "bf16"
QUANT_W8A8_DYNAMIC = "w8a8_dynamic"
SUPPORTED_QUANT_TYPES = (QUANT_BF16, QUANT_W8A8_DYNAMIC)

# Number of independent GEMM instances for L2 cache flush.
# Matches AIConfigurator's outside_loop_count=6 in collect_gemm.py:201.
OUTSIDE_LOOP_COUNT = 6


@dataclass(frozen=True)
class GemmSpec:
    """Immutable GEMM specification."""

    m: int
    n: int
    k: int
    quant_type: str
    dtype: torch.dtype = torch.bfloat16


def _init_vllm_context() -> None:
    """Initialize minimal vLLM global context and distributed environment.

    Mirrors AIConfigurator's setup_distributed() + set_current_vllm_config():
    - Sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT env vars
    - Calls init_distributed_environment()
    - Calls ensure_model_parallel_initialized(1, 1) for single-GPU benchmark
    - Sets current VllmConfig
    """
    import os
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    # Enable NZ (FRACTAL_NZ) format for weight tensors.
    # In production vLLM this is set by vllm_ascend/worker/model_runner_v1.py:156
    # at module import time. Bench scripts don't import the worker layer, so we
    # must set it explicitly before any weight creation.
    import torch
    torch.npu.config.allow_internal_format = True

    init_distributed_environment()
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)

    set_current_vllm_config(VllmConfig())


def _create_single_bf16_gemm(spec: GemmSpec, device: torch.device):
    """Create a single BF16 GEMM instance (AscendRowParallelLinear)."""
    from vllm_ascend.ops.linear import AscendRowParallelLinear

    gemm = AscendRowParallelLinear(
        input_size=spec.k,
        output_size=spec.n,
        bias=False,
        skip_bias_add=True,
        params_dtype=spec.dtype,
        quant_config=None,
        prefix="",
        return_bias=True,
        disable_tp=True,
    )
    gemm.to(device)
    gemm.quant_method.process_weights_after_loading(gemm)
    return gemm


def _create_bf16_gemm(
    spec: GemmSpec, device: torch.device
) -> tuple[Callable[[], None], int]:
    """Create a BF16 GEMM forward function with 6-op L2 cache flush.

    Mirrors AIConfigurator collect_gemm.py:201-208:
      outside_loop_count = 6
      op_list = [create_gemm() for _ in range(6)]
      def kernel_func():
          for op in op_list:
              op.forward(x)
    """
    import torch_npu  # noqa: F401
    import vllm_ascend.patch.worker  # noqa: F401

    op_list = [_create_single_bf16_gemm(spec, device) for _ in range(OUTSIDE_LOOP_COUNT)]
    x = torch.randn(spec.m, spec.k, dtype=spec.dtype, device=device)

    def forward() -> None:
        for op in op_list:
            op.forward(x)

    # Dry run
    forward()
    torch.npu.synchronize()

    return forward, OUTSIDE_LOOP_COUNT


def _create_single_w8a8_gemm(spec: GemmSpec, device: torch.device, qc):
    """Create a single W8A8_DYNAMIC GEMM instance.

    We bypass `process_weights_after_loading` and inline the same logic
    here, because that function's first step (`maybe_trans_nz`) calls
    `torch_npu.npu_format_cast(FRACTAL_NZ)`, which on this CANN install
    fails with `AclSetCompileopt(ACL_OP_JIT_COMPILE) error 500001`
    (no `tbe` python module, AOE InitCannKB fails). When that throws,
    the rest of the function (weight/scale flatten, scale_fp32, offset
    flatten) never runs, leaving the layer in a half-initialised state
    that later trips `aclnnQuantMatmulV4` with `error 161002`
    (scale shape mismatch). Doing it ourselves means we can:
      - try NZ cast and silently fall back to ND on failure
      - still flatten scale/offset and produce weight_scale_fp32
        regardless of whether NZ succeeded
    """
    from vllm_ascend.ops.linear import AscendRowParallelLinear

    gemm = AscendRowParallelLinear(
        input_size=spec.k,
        output_size=spec.n,
        bias=False,
        skip_bias_add=True,
        params_dtype=spec.dtype,
        quant_config=qc,
        prefix="",
        return_bias=True,
        disable_tp=True,
    )
    gemm.to(device)

    with torch.no_grad():
        gemm.weight.data.copy_(
            torch.randint(-128, 127, gemm.weight.shape, dtype=torch.int8, device=device)
        )
        gemm.weight_scale.data.copy_(
            torch.rand(gemm.weight_scale.shape, dtype=spec.dtype, device=device) * 0.1 + 0.01
        )
        gemm.weight_offset.data.zero_()

    # Inline of vllm_ascend AscendW8A8DynamicLinearMethod.process_weights_after_loading:
    #   1. transpose weight (out, in) -> (in, out)
    #   2. cast to FRACTAL_NZ for kernel speed (best-effort; ND also works)
    #   3. flatten scale + offset
    #   4. produce weight_scale_fp32
    with torch.no_grad():
        gemm.weight.data = gemm.weight.data.transpose(0, 1).contiguous()
        try:
            import torch_npu
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            gemm.weight.data = torch_npu.npu_format_cast(
                gemm.weight.data, ACL_FORMAT_FRACTAL_NZ
            )
        except Exception as e:
            # ACL JIT compile / tbe missing — fall back to ND format.
            # aclnnQuantMatmulV4 accepts both NZ and ND for the weight.
            print(f"[WARN] FRACTAL_NZ cast failed, using ND: {type(e).__name__}: {e}")

        gemm.weight_scale.data = gemm.weight_scale.data.flatten()
        gemm.weight_scale_fp32 = gemm.weight_scale.data.to(torch.float32)
        gemm.weight_offset.data = gemm.weight_offset.data.flatten()

    return gemm


def _create_w8a8_dynamic_gemm(
    spec: GemmSpec, device: torch.device
) -> tuple[Callable[[], None], int]:
    """Create a W8A8_DYNAMIC GEMM forward function with 6-op L2 cache flush.

    Bypasses vllm-ascend's AscendW8A8DynamicLinearMethod entirely. The
    upstream linear path (`AscendRowParallelLinear` + the quant method's
    `process_weights_after_loading` + `apply`) hits two ACL problems on
    this CANN install:

      1. process_weights_after_loading -> maybe_trans_nz ->
         npu_format_cast(FRACTAL_NZ) -> AclSetCompileopt(JIT) error
         500001 (no `tbe` python module). Even with the inline
         try/except (see _create_single_w8a8_gemm above), the partially
         executed cast leaves weight storage state that ACL refuses
         later in apply.

      2. apply -> torch_npu.npu_quant_matmul fails with error 161002
         on ND-format weights for some shapes (e.g. logits N=77440,
         dense_gate_up N=24576) — same op/shape works fine when called
         directly with cleanly allocated tensors.

    What we do here instead:
      - Allocate the weight straight in (K, N) layout (the layout
        npu_quant_matmul expects after the transpose step), int8.
      - Allocate weight_scale as a clean 1D bf16 tensor of length N
        (the layout aclnnQuantMatmulV4 expects).
      - Call torch_npu.npu_quant_matmul directly each forward, with
        per-token quant on the input.

    This is exactly what apply() does, just without the long chain of
    layer-state mutations that the upstream code path goes through.
    The benchmark cost is identical because we issue the same kernel.
    """
    import torch_npu  # noqa: F401

    int8_min, int8_max = -128, 127

    def _make_weights() -> tuple[torch.Tensor, torch.Tensor]:
        # weight is (K, N) int8 — already in post-transpose layout
        # the kernel wants. Skip FRACTAL_NZ cast (NZ would be faster
        # but the cast op trips ACL JIT compile here; ND works).
        weight = torch.randint(
            int8_min, int8_max, (spec.k, spec.n),
            dtype=torch.int8, device=device,
        )
        # weight_scale: 1D of length N, bf16. Per-channel.
        weight_scale = (
            torch.rand(spec.n, dtype=spec.dtype, device=device) * 0.1 + 0.01
        )
        return weight, weight_scale

    op_list = [_make_weights() for _ in range(OUTSIDE_LOOP_COUNT)]
    x = torch.randn(spec.m, spec.k, dtype=spec.dtype, device=device)

    def forward() -> None:
        for weight, weight_scale in op_list:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
            need_unsqz = pertoken_scale.dim() == 2
            if need_unsqz:
                quantized_x = quantized_x.squeeze(dim=1)
                pertoken_scale = pertoken_scale.squeeze(dim=1)
            out = torch_npu.npu_quant_matmul(
                quantized_x,
                weight,
                weight_scale,
                pertoken_scale=pertoken_scale,
                bias=None,
                output_dtype=x.dtype,
            )
            if need_unsqz:
                out = out.unsqueeze(dim=1)

    # Dry run
    forward()
    torch.npu.synchronize()

    return forward, OUTSIDE_LOOP_COUNT


def create_gemm_func(
    spec: GemmSpec, device: torch.device
) -> tuple[Callable[[], None], int]:
    """Create a benchmark-ready GEMM function for the given spec.

    Mirrors AIConfigurator's pattern: 6 independent GEMM instances for L2 flush.

    Args:
        spec: GEMM dimensions and quantization type.
        device: Target device (torch.device("npu")).

    Returns:
        Tuple of (forward_func, op_count):
          - forward_func: executes op_count GEMMs in sequence
          - op_count: number of GEMM ops per call (for latency division)

    Raises:
        ValueError: If quant_type is not supported.
    """
    if spec.quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type: {spec.quant_type}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    if spec.quant_type == QUANT_BF16:
        return _create_bf16_gemm(spec, device)
    return _create_w8a8_dynamic_gemm(spec, device)
