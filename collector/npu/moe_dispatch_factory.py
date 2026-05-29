"""Factory for vllm-ascend MoE dispatch+combine bench setup.

Encapsulates everything the probe walked through:
  - NPU / vllm-ascend kernel init (OOT path, _C_ascend.so loading)
  - torch.distributed init (HCCL backend, world from torchrun env vars)
  - vllm config + ParallelConfig (so initialize_model_parallel runs)
  - vllm TP/PP groups (ensure_model_parallel_initialized)
  - vllm-ascend MC2 group (_MC2) — built directly from the ep ranks
  - W8A8 / BF16 weight tensors in FRACTAL_NZ format (npu_format_cast 29)
  - Direct torch.ops._C_ascend.dispatch_ffn_combine call, mirroring the
    canonical path validated by vllm-ascend's nightly tests.

We deliberately bypass FusedMC2CommImpl: its hardcoded
max_output_size=65536 (line 292 of moe_comm_method.py) allocates a
worst-case scratch buffer that OOMs on synthetic single-/double-/16-
card runs. The bare C op accepts max_output_size as a parameter; we
size it to ~max(num_tokens × 2, 512), matching the nightly test
range and approximating what production cudagraph_capture_sizes
gives the kernel.

The collector stays thin: it sets a (num_tokens, ep_size, dtype) spec,
asks the factory for a forward callable + a context to keep alive, and
hands the callable to bench_engine.benchmark_npu().
"""
from __future__ import annotations

import glob
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable

import torch


# ============================================================
# 1. NPU / vllm-ascend bring-up (mirrors collect_mla_module preamble)
# ============================================================
def ensure_oot_custom_opp_path() -> None:
    """Prepend vllm-ascend's OOT vendor dir to ASCEND_CUSTOM_OPP_PATH."""
    if os.environ.get("AIC_SKIP_OOT_PATH_FIX") in {"1", "true", "TRUE"}:
        return
    try:
        import vllm_ascend  # noqa
    except ImportError:
        return
    oot = os.path.join(
        os.path.dirname(vllm_ascend.__file__),
        "_cann_ops_custom", "vendors", "vllm-ascend",
    )
    if not os.path.isdir(oot):
        return
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if oot in existing.split(":"):
        return
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{oot}:{existing}" if existing else oot
    )


def ensure_npu_kernels_loaded() -> None:
    """Force-load vllm_ascend_C.so + register OOT bins."""
    import torch
    torch.npu.config.allow_internal_format = True
    import torch_npu  # noqa
    import vllm_ascend
    from vllm_ascend.platform import NPUPlatform
    NPUPlatform.import_kernels()
    root = os.path.dirname(vllm_ascend.__file__)
    for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so)
        except Exception as e:
            print(f"[WARN] failed to load {so}: {e}", flush=True)


# ============================================================
# 2. torch.distributed + vllm parallel state
# ============================================================
@dataclass(frozen=True)
class DistContext:
    """Bundle of distributed-state handles."""

    rank: int
    world_size: int
    local_rank: int
    ep_world_size: int
    ep_rank: int
    ep_group: torch.distributed.ProcessGroup
    group_name: str   # HCCL comm name string for the dispatch_ffn_combine op


def setup_distributed(ep_world_size: int) -> DistContext:
    """Init torch.distributed (via vllm) + create EP group + name string.

    Reads RANK / WORLD_SIZE / LOCAL_RANK from torchrun's env vars.
    Caller must run inside set_current_vllm_config(...) so vllm's
    initialize_model_parallel can read get_current_vllm_config().
    """
    import torch.distributed as dist
    from vllm.distributed import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size % ep_world_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be a multiple of "
            f"ep_world_size ({ep_world_size})"
        )

    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size, rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank, backend="hccl",
        )

    # vllm TP/PP groups (PrepareAndFinalizeWithMC2 ctor needs _TP).
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    # Build the vllm-ascend MC2 group directly via init_model_parallel_group
    # — this is the canonical path; it creates one ProcessGroup per row
    # of `all_ep_ranks` and stashes the rank's group as _MC2.device_group.
    # We then USE that same group as our ep_group, instead of also calling
    # dist.new_group() which would create a redundant HCCL comm and
    # confuse get_hccl_comm_name (HCCL error 4).
    from vllm.distributed import init_model_parallel_group
    from vllm_ascend.distributed import parallel_state as ps
    all_ep_ranks = [
        list(range(g * ep_world_size, (g + 1) * ep_world_size))
        for g in range(world_size // ep_world_size)
    ]
    backend_name = dist.get_backend()
    if ps._MC2 is None:
        ps._MC2 = init_model_parallel_group(
            all_ep_ranks, local_rank, backend_name, group_name="mc2",
        )
    ep_group = ps._MC2.device_group

    backend = ep_group._get_backend(torch.device("npu"))
    ep_local_rank = dist.get_rank(group=ep_group)
    group_name = backend.get_hccl_comm_name(ep_local_rank)

    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        ep_world_size=ep_world_size,
        ep_rank=ep_local_rank,
        ep_group=ep_group,
        group_name=group_name,
    )


# ============================================================
# 3. vllm config (held for the full sweep)
# ============================================================
def make_vllm_config_context():
    """Return a context manager that puts a minimal VllmConfig in scope.

    Required by initialize_model_parallel and TokenDispatcherWithMC2.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    cfg = VllmConfig()
    try:
        cfg.parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
        )
    except Exception:
        pass
    return set_current_vllm_config(cfg)


# ============================================================
# 4. The bench spec + factory
# ============================================================
SUPPORTED_QUANT_TYPES = ("bf16", "w8a8_dynamic")


@dataclass(frozen=True)
class DispatchSpec:
    """One bench point."""
    num_tokens: int       # M (per-rank tokens)
    hidden: int           # H
    inter: int            # I (per-expert FFN intermediate)
    num_experts: int      # NE (logical experts, global)
    topk: int             # K
    ep_world_size: int    # EP
    quant_type: str       # 'bf16' | 'w8a8_dynamic'

    @property
    def num_local_experts(self) -> int:
        if self.num_experts % self.ep_world_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_world_size ({self.ep_world_size})"
            )
        return self.num_experts // self.ep_world_size


def _make_w8a8_weights(spec: DispatchSpec, dctx: DistContext):
    """Build W8A8 weights as FRACTAL_NZ list[Tensor] (one per local expert).

    Mirrors vllm-ascend nightly test_dispatch_ffn_combine.py:
      - INT8 weights, npu_format_cast(_, 29) -> FRACTAL_NZ
      - INT64 deq_scale via npu_trans_quant_param
      - Per-expert tensors as Python list (not stacked)
    """
    import torch_npu
    dev = f"npu:{dctx.local_rank}"
    NL = spec.num_local_experts
    H, I = spec.hidden, spec.inter

    # Note: weight1 shape is (k, n=2*I) per-expert, weight2 is (k2=I, n2=H).
    # Match nightly test layout: w1=(NL, k, n) with k=H, n=2*I.
    w1_stacked = torch.randint(-16, 16, (NL, H, 2 * I), dtype=torch.int8, device=dev)
    w2_stacked = torch.randint(-16, 16, (NL, I, H),     dtype=torch.int8, device=dev)
    w1_scale_fp32 = torch.rand(NL, 2 * I, dtype=torch.float32, device=dev) * 0.1 + 0.01
    w2_scale_fp32 = torch.rand(NL, H,     dtype=torch.float32, device=dev) * 0.1 + 0.01

    def _trans(per_expert_fp32):
        outs = []
        for i in range(per_expert_fp32.shape[0]):
            outs.append(
                torch_npu.npu_trans_quant_param(
                    per_expert_fp32[i].contiguous(), None,
                ).unsqueeze(0)
            )
        return torch.cat(outs, dim=0).to(torch.int64)

    s1_stacked = _trans(w1_scale_fp32)
    s2_stacked = _trans(w2_scale_fp32)

    w1_list, w2_list, s1_list, s2_list = [], [], [], []
    for i in range(NL):
        w1_list.append(torch_npu.npu_format_cast(w1_stacked[i].contiguous(), 29))
        w2_list.append(torch_npu.npu_format_cast(w2_stacked[i].contiguous(), 29))
        s1_list.append(s1_stacked[i].contiguous())
        s2_list.append(s2_stacked[i].contiguous())

    return w1_list, w2_list, s1_list, s2_list


def _make_bf16_weights(spec: DispatchSpec, dctx: DistContext):
    """Build BF16 weights as FRACTAL_NZ list[Tensor], plus int64 zero scales.

    Mirrors vllm-ascend nightly test_dispatch_ffn_combine_bf16.py:
      - BF16 weights cast to FRACTAL_NZ
      - scale1/scale2 are int64 zeros (kernel still expects them, dtype-agnostic)
    """
    import torch_npu
    dev = f"npu:{dctx.local_rank}"
    NL = spec.num_local_experts
    H, I = spec.hidden, spec.inter

    w1_stacked = torch.randn(NL, H, 2 * I, dtype=torch.bfloat16, device=dev)
    w2_stacked = torch.randn(NL, I, H,     dtype=torch.bfloat16, device=dev)
    s1_stacked = torch.zeros(NL, 2 * I, dtype=torch.int64, device=dev)
    s2_stacked = torch.zeros(NL, H,     dtype=torch.int64, device=dev)

    w1_list, w2_list, s1_list, s2_list = [], [], [], []
    for i in range(NL):
        w1_list.append(torch_npu.npu_format_cast(w1_stacked[i].contiguous(), 29))
        w2_list.append(torch_npu.npu_format_cast(w2_stacked[i].contiguous(), 29))
        s1_list.append(s1_stacked[i].contiguous())
        s2_list.append(s2_stacked[i].contiguous())

    return w1_list, w2_list, s1_list, s2_list


def _resolve_max_output_size(spec: DispatchSpec) -> int:
    """Match production cudagraph_capture_sizes-like upper bound.

    The dispatch_ffn_combine kernel allocates per-rank scratch sized to
    max_output_size × num_local_experts × hidden. vllm-ascend's
    FusedMC2CommImpl wrapper hardcodes max_output_size=65536 (line 292
    of moe_comm_method.py); that fallback ALWAYS OOMs on the synthetic
    bench because we don't have a scheduler bounding output per call.

    The vllm-ascend nightly tests pin max_output_size=512 instead. We
    match that lower bound when num_tokens is small, and grow with
    num_tokens to leave the kernel some headroom on larger M sweeps —
    this matches what production sees: cudagraph_capture_sizes max in
    the GLM-5 deployment is 102 (decode) and max-num-batched-tokens
    is 4096 (prefill).
    """
    return max(spec.num_tokens * 2, 512)


def create_dispatch_combine_func(
    spec: DispatchSpec, dctx: DistContext,
) -> tuple[Callable[[], None], dict]:
    """Build a forward callable that runs one dispatch_ffn_combine pass.

    Bypasses vllm-ascend's FusedMC2CommImpl wrapper and calls
    torch.ops._C_ascend.dispatch_ffn_combine directly, mirroring the
    canonical path validated by the vllm-ascend nightly tests
    (test_dispatch_ffn_combine.py for W8A8, test_dispatch_ffn_combine_bf16.py
    for BF16). The wrapper's hardcoded max_output_size=65536 is what
    OOMs synthetic bench; the bare op accepts any value.

    Returns:
      forward_fn: zero-arg callable, runs one fused dispatch+ffn+combine
      meta: dict with handles to keep alive during benchmarking
    """
    # Make sure vllm-ascend's custom C ops are registered
    # (torch.ops._C_ascend.dispatch_ffn_combine).
    from vllm_ascend.utils import enable_custom_op
    enable_custom_op()

    M = spec.num_tokens
    H = spec.hidden
    NL = spec.num_local_experts
    NE = spec.num_experts
    K = spec.topk
    dev = f"npu:{dctx.local_rank}"

    if spec.quant_type == "w8a8_dynamic":
        out_dtype = torch.bfloat16
        w1_list, w2_list, s1_list, s2_list = _make_w8a8_weights(spec, dctx)
    elif spec.quant_type == "bf16":
        out_dtype = torch.bfloat16
        w1_list, w2_list, s1_list, s2_list = _make_bf16_weights(spec, dctx)
    else:
        raise ValueError(f"unsupported quant_type: {spec.quant_type}")

    x = torch.randn(M, H, dtype=out_dtype, device=dev)
    expert_idx = torch.randint(0, NE, (M, K), dtype=torch.int32, device=dev)
    probs = torch.rand(M, K, dtype=torch.float32, device=dev)

    out = torch.empty_like(x)
    expert_token_nums = torch.zeros((1, NL), dtype=torch.int32, device=dev)

    max_output_size = _resolve_max_output_size(spec)

    def forward_fn() -> None:
        torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore[attr-defined]
            x=x,
            weight1=w1_list,
            weight2=w2_list,
            expert_idx=expert_idx,
            scale1=s1_list,
            scale2=s2_list,
            probs=probs,
            group=dctx.group_name,
            max_output_size=max_output_size,
            out=out,
            expert_token_nums=expert_token_nums,
        )

    # Dry run to surface init failures here, not in the timing loop.
    forward_fn()
    torch.npu.synchronize()

    return forward_fn, {
        "x": x, "out": out, "expert_idx": expert_idx, "probs": probs,
        "w1": w1_list, "w2": w2_list, "s1": s1_list, "s2": s2_list,
        "expert_token_nums": expert_token_nums,
        "max_output_size": max_output_size,
    }


# ============================================================
# 5. One-call setup for the collector entry point
# ============================================================
def setup_all(ep_world_size: int) -> tuple[DistContext, ExitStack]:
    """One-line bring-up: returns (DistContext, exit_stack).

    Caller must call exit_stack.close() at the end.
    """
    ensure_oot_custom_opp_path()
    ensure_npu_kernels_loaded()

    stack = ExitStack()
    stack.enter_context(make_vllm_config_context())
    dctx = setup_distributed(ep_world_size)
    return dctx, stack
