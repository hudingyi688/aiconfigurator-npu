"""MLA operator factory for Ascend NPU benchmarking.

Directly invokes the same torch_npu kernels as
vllm_ascend.attention.mla_v1.AscendMLAImpl:
  - Decode:  npu_fused_infer_attention_score_v2  (BNSD_NBSD layout)
  - Prefill: npu_fused_infer_attention_score     (TND layout)
"""
import logging
import math
from dataclasses import dataclass
from typing import Callable
from unittest.mock import MagicMock
import torch

try:
    import torch_npu
except ImportError:
    pass

logger = logging.getLogger(__name__)

OP_CONTEXT = "mla_context"
OP_GENERATION = "mla_generation"
SUPPORTED_OP_TYPES = (OP_CONTEXT, OP_GENERATION)

BLOCK_SIZE = 128


@dataclass(frozen=True)
class MlaSpec:
    """Immutable MLA benchmark specification (DeepSeek-V3 defaults)."""
    op_type: str
    batch: int
    seq_len: int
    num_heads: int        # 128
    kv_lora_rank: int     # 512
    qk_nope_head_dim: int # 128
    qk_rope_head_dim: int # 64
    v_head_dim: int       # 128
    dtype: torch.dtype = torch.bfloat16

    @property
    def head_size(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_head_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dry_run(forward: Callable, spec: MlaSpec, phase: str) -> None:
    """Execute once to catch shape / tiling errors early."""
    try:
        forward()
        torch.npu.synchronize()
    except Exception as e:
        try:
            torch.npu.synchronize()
        except RuntimeError:
            pass
        raise RuntimeError(
            f"{phase} dry-run failed for "
            f"batch={spec.batch} seq={spec.seq_len} "
            f"heads={spec.num_heads}: {e}"
        ) from e


_forward_ctx_initialized = False
_ctx_refs = []

def _ensure_forward_context() -> None:
    global _forward_ctx_initialized
    if _forward_ctx_initialized:
        return
    try:
        from vllm.config import VllmConfig, set_current_vllm_config, ModelConfig
        from vllm.forward_context import set_forward_context
        from vllm_ascend.ascend_config import init_ascend_config

        vllm_config = VllmConfig()
        # Create minimal nested configs explicitly requested by AscendMLAImpl
        class MockConfig: pass
        model_config = MockConfig()
        model_config.dtype = torch.bfloat16
        
        speculative_config = MockConfig()
        speculative_config.num_speculative_tokens = 4
        speculative_config.disable_padded_drafter_batch = False
        
        parallel_config = MockConfig()
        parallel_config.prefill_context_parallel_size = 1
        
        quant_config = MockConfig()
        quant_config.enabling_fa_quant = lambda *args: False
        
        vllm_config.model_config = model_config # type: ignore
        vllm_config.speculative_config = speculative_config # type: ignore
        vllm_config.parallel_config = parallel_config # type: ignore
        vllm_config.quant_config = quant_config # type: ignore
        vllm_config.kv_transfer_config = None # type: ignore
        vllm_config.additional_config = {} # type: ignore
        
        init_ascend_config(vllm_config)
        cfg_ctx = set_current_vllm_config(vllm_config)
        cfg_ctx.__enter__()
        _ctx_refs.append(cfg_ctx)

        fwd_ctx = set_forward_context(attn_metadata=None, vllm_config=vllm_config)
        fwd_ctx.__enter__()
        _ctx_refs.append(fwd_ctx)

        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        ctx.capturing = False
        _forward_ctx_initialized = True
    except ImportError as e:
        logger.warning(f"Failed to initialize proper vllm context, depending on raw mocks: {e}")


def _create_impl(spec: MlaSpec, device: torch.device):
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl
    
    kv_a_layernorm = MagicMock()
    kv_a_layernorm.weight = torch.randn(spec.kv_cache_head_size, device=device, dtype=spec.dtype)
    kv_a_layernorm.variance_epsilon = 1e-6

    kwargs = {
        "kv_lora_rank": spec.kv_lora_rank,
        "qk_nope_head_dim": spec.qk_nope_head_dim,
        "qk_rope_head_dim": spec.qk_rope_head_dim,
        "qk_head_dim": spec.head_size,
        "v_head_dim": spec.v_head_dim,
        "q_lora_rank": 1536,
        "q_proj": MagicMock(),
        "q_b_proj": MagicMock(),
        "kv_b_proj": MagicMock(),
        "o_proj": MagicMock(),
        "kv_a_proj_with_mqa": MagicMock(),
        "fused_qkv_a_proj": MagicMock(),
        "kv_a_layernorm": kv_a_layernorm,
        "rotary_emb": MagicMock(),
        "layer_name": "mock_layer",
    }

    # By passing kwargs directly mirroring test_mla_v1.py, we invoke AscendMLAImpl.
    impl = AscendMLAImpl(
        num_heads=spec.num_heads,
        head_size=spec.head_size,
        scale=1.0 / math.sqrt(spec.head_size),
        num_kv_heads=1,  # MLA has 1 latent KV cache head
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        blocksparse_params=None,
        logits_soft_cap=None,
        attn_type=None,
        kv_sharing_target_layer_name=None,
        **kwargs
    )
    impl.fa_quant_layer = False
    
    # Mock linear projections dynamically called by the kernel implementation
    def dummy_v_up_proj(x): return x
    impl._v_up_proj = dummy_v_up_proj
    
    return impl


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _create_generation_mla(
    spec: MlaSpec, device: torch.device
) -> Callable[[], torch.Tensor]:
    _ensure_forward_context()
    impl = _create_impl(spec, device)
    
    num_heads = spec.num_heads
    num_kv_heads = 1
    block_size = BLOCK_SIZE

    blocks_per_seq = math.ceil(spec.seq_len / block_size)
    num_blocks = spec.batch * blocks_per_seq

    q_nope = torch.randn(spec.batch, num_heads, spec.kv_lora_rank, dtype=spec.dtype, device=device)
    q_pe = torch.randn(spec.batch, num_heads, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)

    # Note: _forward_decode applies specific reshape formats, we feed it flat layout
    k_nope = torch.randn(num_blocks * num_kv_heads * block_size, spec.kv_lora_rank, dtype=spec.dtype, device=device)
    k_pe = torch.randn(num_blocks * num_kv_heads * block_size, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, AscendMLADecodeMetadata
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(spec.batch, blocks_per_seq)
    
    decode_metadata = AscendMLADecodeMetadata(
        input_positions=torch.zeros(spec.batch, dtype=torch.int32, device=device),
        block_table=block_table,
        seq_lens=torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device),
        max_seq_lens=spec.seq_len,
        seq_lens_list=[spec.seq_len] * spec.batch,
        actual_seq_lengths_q=[1] * spec.batch,
        attn_mask=None,
        sin=None,
        cos=None,
        cp_seq_len=None
    )

    metadata = AscendMLAMetadata(
        num_actual_tokens_pcp_padded=spec.batch,
        num_actual_tokens=spec.batch,
        slot_mapping=torch.zeros(spec.batch, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(0, spec.batch + 1, dtype=torch.int32, device=device),
        seq_lens=decode_metadata.seq_lens,
        block_tables=block_table,
        num_decodes=spec.batch,
        num_decode_tokens=spec.batch,
        num_prefills=0,
        attn_state=AscendAttentionState.DecodeOnly,
        decode=decode_metadata
    )

    def forward() -> torch.Tensor:
        return impl._forward_decode(q_nope, q_pe, k_nope, k_pe, block_size, metadata)

    _dry_run(forward, spec, "mla_decode_v1_impl")
    return forward


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------

def _create_context_mla(
    spec: MlaSpec, device: torch.device
) -> Callable[[], torch.Tensor]:
    _ensure_forward_context()
    impl = _create_impl(spec, device)
    
    num_tokens = spec.batch * spec.seq_len

    q_nope = torch.randn(num_tokens, spec.num_heads, spec.qk_nope_head_dim, dtype=spec.dtype, device=device)
    q_pe = torch.randn(num_tokens, spec.num_heads, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)
    k_nope = torch.randn(num_tokens, spec.num_heads, spec.qk_nope_head_dim, dtype=spec.dtype, device=device)
    k_pe = torch.randn(num_tokens, spec.num_heads, spec.qk_rope_head_dim, dtype=spec.dtype, device=device)
    value = torch.randn(num_tokens, spec.num_heads, spec.v_head_dim, dtype=spec.dtype, device=device)
    kv_cache = (k_nope, k_pe) # Dummy to bypass assertions

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata, AscendMLAPrefillMetadata
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    actual_seq_lengths_q = [(i + 1) * spec.seq_len for i in range(spec.batch)]
    
    attn_mask = torch.triu(torch.ones(spec.seq_len, spec.seq_len, dtype=torch.int8, device=device), diagonal=1)

    prefill_metadata = AscendMLAPrefillMetadata(
        attn_mask=attn_mask,
        query_lens=torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device),
        seq_lens=[spec.seq_len] * spec.batch,
        context_lens=torch.zeros(spec.batch, dtype=torch.int32, device=device),
        input_positions=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(0, num_tokens + 1, spec.seq_len, dtype=torch.int32, device=device),
        block_table=torch.zeros((spec.batch, 1), dtype=torch.int32, device=device),
        max_query_len=spec.seq_len,
        max_seq_lens=spec.seq_len,
        actual_seq_lengths_q=actual_seq_lengths_q
    )

    metadata = AscendMLAMetadata(
        num_actual_tokens_pcp_padded=num_tokens,
        num_actual_tokens=num_tokens,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        query_start_loc=prefill_metadata.query_start_loc,
        seq_lens=torch.tensor([spec.seq_len] * spec.batch, dtype=torch.int32, device=device),
        block_tables=prefill_metadata.block_table,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=spec.batch,
        attn_state=AscendAttentionState.PrefillNoCache,
        prefill=prefill_metadata
    )

    def forward() -> torch.Tensor:
        # returns output, lse. Fetch [0]
        return impl._forward_prefill(q_nope, q_pe, k_nope, k_pe, value, kv_cache, metadata)

    _dry_run(forward, spec, "mla_prefill_v1_impl")
    return forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_mla_func(
    spec: MlaSpec, device: torch.device
) -> Callable[[], tuple]:
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type}")

    if spec.op_type == OP_CONTEXT:
        return _create_context_mla(spec, device)
    return _create_generation_mla(spec, device)
