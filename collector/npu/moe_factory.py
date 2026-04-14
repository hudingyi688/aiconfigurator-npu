"""MoE operator factory for Ascend NPU benchmarking.

Constructs MoE operators via vllm-ascend's framework-level interfaces,
ensuring the benchmark exercises the same kernel path as real inference.

Two paths:
  - BF16: unquant_apply_mlp(need_trans=False) → npu_grouped_matmul + npu_swiglu
  - W8A8_DYNAMIC: quant_apply_mlp() → npu_dynamic_quant + npu_grouped_matmul_swiglu_quant
                                       + npu_grouped_matmul_gmm2

Full MoE pipeline on NPU (matching AllGatherCommImpl.fused_experts):
  1. Token dispatch: PyTorch argsort (pre-computed, not timed)
  2. MLP compute: unquant_apply_mlp / quant_apply_mlp (vllm-ascend framework)
  3. Token combine: npu_moe_token_unpermute (CANN kernel)

Call chain alignment with AIConfigurator:
  AIConfigurator (GPU):  fused_experts(hidden, w1, w2, topk_weights, topk_ids, ...)
  NPU (this script):     routing → apply_mlp → token_unpermute
  Both call the framework-level MoE API, not raw kernels.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Quantization type constants
QUANT_BF16 = "bf16"
QUANT_W8A8_DYNAMIC = "w8a8_dynamic"
SUPPORTED_QUANT_TYPES = (QUANT_BF16, QUANT_W8A8_DYNAMIC)


@dataclass(frozen=True)
class MoeSpec:
    """Immutable MoE benchmark specification."""

    num_tokens: int
    hidden_size: int
    intermediate_size: int  # per-expert FFN intermediate dim
    num_experts: int
    topk: int
    quant_type: str
    ep_size: int = 1  # simulated EP: only allocate num_experts // ep_size local experts
    dtype: torch.dtype = torch.bfloat16

    @property
    def local_num_experts(self) -> int:
        return self.num_experts // self.ep_size


_forward_context_initialized = False


def _ensure_forward_context(num_tokens: int = 1) -> None:
    """Set up minimal vLLM + vllm-ascend forward context for MoE ops.

    quant_apply_mlp / unquant_apply_mlp access _EXTRA_CTX.moe_comm_type
    and DeviceOperator.npu_moe_init_routing accesses forward_context.num_tokens,
    both requiring an active forward context.

    We directly set the _forward_context global to avoid issues with
    set_forward_context's signature varying across vllm versions.
    """
    global _forward_context_initialized
    import vllm.forward_context as _fc_module

    if not _forward_context_initialized:
        _forward_context_initialized = True

        # Enable NZ (FRACTAL_NZ) format for weight tensors.
        # In production vLLM this is set by vllm_ascend/worker/model_runner_v1.py:156.
        torch.npu.config.allow_internal_format = True

        from vllm.config import VllmConfig, set_current_vllm_config

        # Keep vllm_config alive for the process
        _vllm_config = VllmConfig()
        _cfg_ctx = set_current_vllm_config(_vllm_config)
        _cfg_ctx.__enter__()

        # Load vllm-ascend custom C++ ops (moe_grouped_matmul etc.)
        try:
            from vllm_ascend.utils import enable_custom_op
            enable_custom_op()
        except Exception:
            pass

        # Initialize triton device properties (needed by swiglu_quant in W8A8)
        try:
            from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
            init_device_properties_triton()
        except Exception:
            pass

        # Create a minimal ForwardContext object directly.
        # ForwardContext is typically a dataclass or simple object;
        # we create one via set_forward_context's yield, or build manually.
        try:
            from vllm.forward_context import ForwardContext
            ctx = ForwardContext(
                no_compile_layers=set(),
                attn_metadata=None,
                vllm_config=_vllm_config,
                virtual_engine=0,
                num_tokens=num_tokens,
            )
        except (ImportError, TypeError):
            # Fallback: use a simple namespace object
            class _MinimalForwardContext:
                pass
            ctx = _MinimalForwardContext()
            ctx.no_compile_layers = set()
            ctx.attn_metadata = None
            ctx.vllm_config = _vllm_config
            ctx.virtual_engine = 0
            ctx.dp_metadata = None
            ctx.skip_compiled = False

        # Set it as the global forward context
        _fc_module._forward_context = ctx

        # Set vllm-ascend extra attributes
        from vllm_ascend.ascend_forward_context import MoECommType
        ctx.moe_comm_type = MoECommType.ALLGATHER
        ctx.moe_comm_method = None
        ctx.in_profile_run = False

    # Always update num_tokens for the current spec
    ctx = _fc_module._forward_context
    ctx.num_tokens = num_tokens


def _generate_routing(
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random routing decisions (topk_ids, topk_weights).

    Uses uniform random logits → topk selection → softmax normalization.
    Simpler than AIConfigurator's Power Law distribution but sufficient
    for kernel-level benchmarking.
    """
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(logits, k=topk, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
    return topk_ids.to(torch.int32), topk_weights


def _pytorch_token_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch token dispatch — keeps ALL num_tokens*topk rows.

    Does NOT filter by local experts. All tokens are kept and sorted by
    expert ID so that local experts (lower IDs) come first. group_list
    has num_experts entries; the caller slices to local_num_experts.
    npu_moe_token_unpermute requires permuted_tokens.shape[0] ==
    probs.shape[0] * probs.shape[1], so we must not drop any rows.
    """
    num_tokens, hidden_size = hidden_states.shape

    # Expand hidden: [T, H] -> [T*K, H]
    expanded_hidden = hidden_states.unsqueeze(1).expand(
        -1, topk, -1,
    ).reshape(-1, hidden_size)

    # Flatten expert ids: [T, K] -> [T*K]
    flat_expert_ids = topk_ids.reshape(-1).long()

    # Sort by expert ID (stable sort keeps token order within each expert)
    sort_order = torch.argsort(flat_expert_ids, stable=True)
    sorted_hidden = expanded_hidden[sort_order]
    sorted_expert_ids = flat_expert_ids[sort_order]
    expanded_row_idx = sort_order.to(torch.int32)

    # Per-expert token counts
    group_list = torch.zeros(
        num_experts, dtype=torch.int64, device=hidden_states.device,
    )
    if sorted_expert_ids.numel() > 0:
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True,
        )
        group_list[unique_experts] = counts

    return sorted_hidden, expanded_row_idx, group_list


def _token_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_num_experts: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token dispatch — keeps all num_tokens*topk rows, slices group_list.

    All tokens are kept (no EP filtering) so that sorted_hidden.shape[0]
    == num_tokens * topk, which npu_moe_token_unpermute requires.
    group_list is sliced to local_num_experts for MLP compute.
    """
    sorted_hidden, expanded_row_idx, group_list = _pytorch_token_dispatch(
        hidden_states, topk_ids, num_experts, topk,
    )
    if local_num_experts < num_experts:
        group_list = group_list[:local_num_experts]
    return sorted_hidden, group_list, expanded_row_idx, topk_weights


def _token_combine(
    hidden_states: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Token combine via npu_moe_token_unpermute.

    Mirrors TokenDispatcherWithAllGather.token_combine() — uses
    npu_moe_token_unpermute (CANN kernel) for the real inference path.
    """
    import torch_npu  # noqa: F401

    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )


def _create_bf16_moe(
    spec: MoeSpec, device: torch.device,
) -> Callable[[], torch.Tensor]:
    """Create a BF16 (unquantized) MoE forward function.

    Calls unquant_apply_mlp() — the vllm-ascend framework-level BF16 MoE MLP.
    Internally: npu_grouped_matmul(GMM1) → npu_swiglu → npu_grouped_matmul(GMM2).

    Weight layout: [E, in_dim, out_dim] — same as after process_weights_after_loading.
    need_trans=False (default in inference) skips the internal transpose.

    Pipeline: routing → unquant_apply_mlp(w1, w2) → pad → npu_moe_token_unpermute
    """
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp

    _ensure_forward_context(num_tokens=spec.num_tokens)
    local_experts = spec.local_num_experts

    # Weights: [local_num_experts, in_dim, out_dim]
    # Same layout as after process_weights_after_loading (already transposed).
    # need_trans=False in inference — no transpose inside unquant_apply_mlp.
    w1 = torch.randn(
        local_experts, spec.hidden_size, spec.intermediate_size * 2,
        dtype=spec.dtype, device=device,
    )
    w2 = torch.randn(
        local_experts, spec.intermediate_size, spec.hidden_size,
        dtype=spec.dtype, device=device,
    )

    # Input hidden states
    hidden_states = torch.randn(
        spec.num_tokens, spec.hidden_size, dtype=spec.dtype, device=device,
    )

    # Pre-compute routing
    topk_ids, topk_weights = _generate_routing(
        spec.num_tokens, spec.num_experts, spec.topk, device,
    )

    # Pre-compute token dispatch (keeps all num_tokens*topk rows)
    sorted_hidden, group_list, expanded_row_idx, topk_weights = _token_dispatch(
        hidden_states, topk_ids, topk_weights,
        spec.num_experts, local_experts, spec.topk,
    )
    torch.npu.synchronize()

    # Number of tokens processed by MLP (only local experts)
    local_token_count = int(group_list.sum().item())
    total_token_count = spec.num_tokens * spec.topk

    def forward() -> torch.Tensor:
        # MLP compute via vllm-ascend framework (only local expert tokens)
        mlp_input = sorted_hidden[:local_token_count]
        mlp_out = unquant_apply_mlp(
            hidden_states=mlp_input,
            w1=w1,
            w2=w2,
            group_list=group_list,
            group_list_type=1,
            need_trans=False,
        )

        # Pad MLP output back to num_tokens*topk rows for token_unpermute
        if local_token_count < total_token_count:
            padded = torch.zeros(
                total_token_count, mlp_out.shape[1],
                dtype=mlp_out.dtype, device=mlp_out.device,
            )
            padded[:local_token_count] = mlp_out
        else:
            padded = mlp_out

        return _token_combine(padded, expanded_row_idx, topk_weights)

    # Dry run
    forward()
    torch.npu.synchronize()

    return forward


def _create_w8a8_dynamic_moe(
    spec: MoeSpec, device: torch.device,
) -> Callable[[], torch.Tensor]:
    """Create a W8A8_DYNAMIC (INT8) MoE forward function.

    Calls quant_apply_mlp() — the framework-level quantized MoE MLP.
    With moe_comm_type=ALLGATHER (non-MC2), it enters the else branch
    (moe_mlp.py:241-319) which calls:
      npu_dynamic_quant → npu_grouped_matmul(w1) + swiglu_quant
      → npu_grouped_matmul_gmm2(w2)

    Weight format: w1/w2 as list[Tensor], w1_scale/w2_scale as list[Tensor],
    matching the npu_grouped_matmul(weight=w1, scale=w1_scale) call pattern.

    Pipeline: init_routing → quant_apply_mlp(w1, w2, scales) → token_unpermute
    """
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp

    _ensure_forward_context(num_tokens=spec.num_tokens)
    local_experts = spec.local_num_experts

    # W8A8 weights: [local_num_experts, in_dim, out_dim] in int8
    # Same layout as BF16: npu_grouped_matmul computes x @ w (no transpose)
    # NZ format only changes physical memory layout, not logical shape
    w1 = torch.randint(
        -128, 127,
        (local_experts, spec.hidden_size, spec.intermediate_size * 2),
        dtype=torch.int8, device=device,
    )
    w2 = torch.randint(
        -128, 127,
        (local_experts, spec.intermediate_size, spec.hidden_size),
        dtype=torch.int8, device=device,
    )

    # NZ format conversion (same as W8A8 process_weights_after_loading)
    w1.data = torch_npu.npu_format_cast(w1.data, 29)  # ACL_FORMAT_FRACTAL_NZ = 29
    w2.data = torch_npu.npu_format_cast(w2.data, 29)

    # Per-channel weight scales
    # IMPORTANT: w2_scale dtype determines output_dtype in quant_apply_mlp
    # (line: _output_dtype = w2_scale[0].dtype), so use bfloat16 not float32
    w1_scale = (
        torch.rand(local_experts, spec.intermediate_size * 2,
                    dtype=torch.bfloat16, device=device) * 0.1 + 0.01
    )
    w2_scale = (
        torch.rand(local_experts, spec.hidden_size,
                    dtype=torch.bfloat16, device=device) * 0.1 + 0.01
    )

    # Input hidden states
    hidden_states = torch.randn(
        spec.num_tokens, spec.hidden_size, dtype=spec.dtype, device=device,
    )

    # Pre-compute routing
    topk_ids, topk_weights = _generate_routing(
        spec.num_tokens, spec.num_experts, spec.topk, device,
    )

    # Pre-compute token dispatch
    sorted_hidden, group_list, expanded_row_idx, topk_weights = _token_dispatch(
        hidden_states, topk_ids, topk_weights,
        spec.num_experts, local_experts, spec.topk,
    )
    torch.npu.synchronize()

    # Number of tokens processed by MLP (only local experts)
    local_token_count = int(group_list.sum().item())
    total_token_count = spec.num_tokens * spec.topk

    def forward() -> torch.Tensor:
        # Clone because quant_apply_mlp calls dispose_tensor() on input
        mlp_input = sorted_hidden[:local_token_count].clone()
        mlp_out = quant_apply_mlp(
            hidden_states=mlp_input,
            w1=[w1],
            w1_scale=[w1_scale],
            w2=[w2],
            w2_scale=[w2_scale],
            group_list=group_list,
            group_list_type=1,
        )

        # Pad MLP output back to num_tokens*topk rows for token_unpermute
        if local_token_count < total_token_count:
            padded = torch.zeros(
                total_token_count, mlp_out.shape[1],
                dtype=mlp_out.dtype, device=mlp_out.device,
            )
            padded[:local_token_count] = mlp_out
        else:
            padded = mlp_out

        return _token_combine(padded, expanded_row_idx, topk_weights)

    # Dry run
    forward()
    torch.npu.synchronize()

    return forward


def create_moe_func(
    spec: MoeSpec, device: torch.device,
) -> Callable[[], torch.Tensor]:
    """Create a benchmark-ready MoE function for the given spec.

    Args:
        spec: MoE dimensions and quantization type.
        device: Target device (torch.device("npu")).

    Returns:
        Zero-arg callable that executes one full MoE forward pass.

    Raises:
        ValueError: If quant_type is not supported.
    """
    if spec.quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type: {spec.quant_type}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    if spec.quant_type == QUANT_BF16:
        return _create_bf16_moe(spec, device)
    return _create_w8a8_dynamic_moe(spec, device)
