"""Attention operator factory for Ascend NPU benchmarking.

Constructs attention operators via vllm-ascend's AscendAttentionBackendImpl,
ensuring the benchmark exercises the same code path as real vLLM inference.

Mirrors AIconfigurator's approach: instantiate vLLM attention impl, construct
metadata, call impl.forward() — not direct torch_npu kernel calls.

Two paths:
  - Context (Prefill): PrefillNoCache state -> FIA kernel
  - Generation (Decode): DecodeOnly state -> FIA kernel with paged KV cache

vllm-ascend interfaces used:
  - AscendAttentionBackendImpl (instantiation + forward)
  - AscendMetadata (PrefillNoCache / DecodeOnly states)
  - AscendAttentionState (state enum)
  - DeviceOperator.reshape_and_cache (called inside impl.forward for decode)
  - vllm.forward_context.set_forward_context (global context for _EXTRA_CTX)
  - vllm.config.VllmConfig / set_current_vllm_config (global vLLM config)
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)

# Op type constants
OP_CONTEXT = "attention_context"
OP_GENERATION = "attention_generation"
SUPPORTED_OP_TYPES = (OP_CONTEXT, OP_GENERATION)

# Ascend fixed block size for paged KV cache
BLOCK_SIZE = 128


@dataclass(frozen=True)
class AttnSpec:
    """Immutable attention benchmark specification."""

    op_type: str
    batch: int
    seq_len: int
    num_heads: int
    num_kv_heads: int  # 0 = MHA (internally resolved to num_heads)
    head_size: int = 128
    dtype: torch.dtype = torch.bfloat16


def _resolve_kv_heads(spec: AttnSpec) -> int:
    """Resolve num_kv_heads: 0 means MHA (same as num_heads)."""
    return spec.num_heads if spec.num_kv_heads == 0 else spec.num_kv_heads


def _generate_causal_mask(device: torch.device) -> torch.Tensor:
    """Generate fixed 2048x2048 causal mask matching vllm-ascend FIA requirement."""
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)


class _MockAttentionLayer:
    """Minimal mock of vLLM AttentionLayer for impl.forward()."""

    def __init__(self) -> None:
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


# ── vllm-ascend interface: AscendAttentionBackendImpl ──

def _create_impl(spec: AttnSpec, device: torch.device):
    """Instantiate AscendAttentionBackendImpl via vllm-ascend."""
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    num_kv_heads = _resolve_kv_heads(spec)
    return AscendAttentionBackendImpl(
        num_heads=spec.num_heads,
        head_size=spec.head_size,
        scale=1.0 / math.sqrt(spec.head_size),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
    )


# ── vllm-ascend interface: AscendMetadata construction ──

def _build_prefill_metadata(spec: AttnSpec, device: torch.device):
    """Build AscendMetadata for PrefillNoCache state."""
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata

    num_tokens = spec.batch * spec.seq_len
    actual_seq_q = (torch.arange(1, spec.batch + 1, dtype=torch.int32) * spec.seq_len).tolist()
    seq_lens = torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, num_tokens + 1, spec.seq_len, dtype=torch.int32, device=device)

    return AscendMetadata(
        attn_state=AscendAttentionState.PrefillNoCache,
        attn_mask=_generate_causal_mask(device),
        num_actual_tokens=num_tokens,
        seq_lens=seq_lens,
        seq_lens_list=[spec.seq_len] * spec.batch,
        actual_seq_lengths_q=actual_seq_q,
        query_start_loc=query_start_loc,
        max_query_len=spec.seq_len,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        causal=True,
        model_runner_type="",
    )


def _build_decode_metadata(spec: AttnSpec, device: torch.device, blocks_per_seq: int):
    """Build AscendMetadata for DecodeOnly state."""
    from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata

    num_blocks = spec.batch * blocks_per_seq
    seq_lens = torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device)
    actual_seq_q = list(range(1, spec.batch + 1))
    query_start_loc = torch.arange(0, spec.batch + 1, dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(spec.batch, blocks_per_seq)
    slot_mapping = torch.zeros(spec.batch, dtype=torch.int32, device=device)

    return AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        attn_mask=_generate_causal_mask(device),
        num_actual_tokens=spec.batch,
        seq_lens=seq_lens,
        seq_lens_list=[spec.seq_len] * spec.batch,
        actual_seq_lengths_q=actual_seq_q,
        query_start_loc=query_start_loc,
        max_query_len=1,
        block_tables=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        model_runner_type="",
    )


# ── vllm interface: forward context ──

_forward_ctx_initialized = False
_ctx_refs = []  # keep context manager references alive


def _ensure_forward_context() -> None:
    """Set up vLLM global config + forward context (once).

    Must be called before any AscendAttentionBackendImpl instantiation,
    as __init__ calls get_current_vllm_config().
    """
    global _forward_ctx_initialized
    if _forward_ctx_initialized:
        return

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    vllm_config = VllmConfig()

    # Enter set_current_vllm_config context and keep alive
    cfg_ctx = set_current_vllm_config(vllm_config)
    cfg_ctx.__enter__()
    _ctx_refs.append(cfg_ctx)

    # Enter set_forward_context and keep alive
    # This sets the thread-local _forward_context so get_forward_context() works
    fwd_ctx = set_forward_context(attn_metadata=None, vllm_config=vllm_config)
    fwd_ctx.__enter__()
    _ctx_refs.append(fwd_ctx)

    # Verify it worked
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    # Set capturing=False explicitly
    ctx.capturing = False

    _forward_ctx_initialized = True


def _dry_run(forward: Callable, spec: AttnSpec, phase: str, num_kv_heads: int) -> None:
    """Dry run with error recovery to avoid poisoning the NPU stream."""
    try:
        forward()
        torch.npu.synchronize()
    except RuntimeError as e:
        try:
            torch.npu.synchronize()
        except RuntimeError:
            pass
        raise RuntimeError(
            f"impl.forward() {phase} dry run failed for "
            f"batch={spec.batch} seq={spec.seq_len} "
            f"heads={spec.num_heads} kv_heads={num_kv_heads}: {e}"
        ) from e


# ── Factory functions ──

def _create_context_attn(spec: AttnSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    """Context (prefill) via AscendAttentionBackendImpl.forward(), PrefillNoCache."""
    _ensure_forward_context()
    num_kv_heads = _resolve_kv_heads(spec)
    num_tokens = spec.batch * spec.seq_len

    impl = _create_impl(spec, device)
    layer = _MockAttentionLayer()
    metadata = _build_prefill_metadata(spec, device)

    query = torch.randn(num_tokens, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    kv_cache = ()
    output = torch.empty(num_tokens, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)

    def forward() -> torch.Tensor:
        return impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    _dry_run(forward, spec, "prefill", num_kv_heads)
    return forward


def _create_generation_attn(spec: AttnSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    """Generation (decode) via AscendAttentionBackendImpl.forward(), DecodeOnly."""
    _ensure_forward_context()
    num_kv_heads = _resolve_kv_heads(spec)
    blocks_per_seq = math.ceil(spec.seq_len / BLOCK_SIZE)
    num_blocks = spec.batch * blocks_per_seq

    impl = _create_impl(spec, device)
    layer = _MockAttentionLayer()
    metadata = _build_decode_metadata(spec, device, blocks_per_seq)

    query = torch.randn(spec.batch, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)
    key = torch.randn(spec.batch, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    value = torch.randn(spec.batch, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    value_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, spec.head_size, dtype=spec.dtype, device=device)
    kv_cache = (key_cache, value_cache)
    output = torch.empty(spec.batch, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)

    def forward() -> torch.Tensor:
        return impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    _dry_run(forward, spec, "decode", num_kv_heads)
    return forward


def create_attn_func(spec: AttnSpec, device: torch.device) -> Callable[[], torch.Tensor]:
    """Create a benchmark-ready attention function via AscendAttentionBackendImpl.forward()."""
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type}. Supported: {SUPPORTED_OP_TYPES}")

    num_kv_heads = _resolve_kv_heads(spec)
    if spec.num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({spec.num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    if spec.op_type == OP_CONTEXT:
        return _create_context_attn(spec, device)
    return _create_generation_attn(spec, device)
