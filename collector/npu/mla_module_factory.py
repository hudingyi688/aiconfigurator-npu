"""DSA module factory for Ascend NPU benchmarking (Method C — Module level).

Constructs and returns a callable that runs the complete
DeepseekV2MLAAttention.forward() pass (projections + DSA attention + output).

Calling chain (GLM-5 on NPU):
  create_dsa_module_func()
    → DeepseekV2MLAAttention.forward()
      → fused_qkv_a_proj → q_a_layernorm → q_b_proj → Q
      → kv_a_proj → kv_a_layernorm → kv_c + k_pe
      → AscendSFAImpl.forward()   ← DSA backend (use_sparse=True)
        → indexer_select_pre_process / post_process
        → npu_sparse_flash_attention
      → o_proj → output
"""

import math
import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context

# Patch config registry so AutoConfig resolves glm_moe_dsa → DeepseekV3Config.
try:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    if "glm_moe_dsa" not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════
# Spec
# ═══════════════════════════════════════════════════════════════════════

OP_CONTEXT = "dsa_context"
OP_GENERATION = "dsa_generation"
SUPPORTED_OP_TYPES = (OP_CONTEXT, OP_GENERATION)


@dataclass(frozen=True)
class DsaModuleSpec:
    """Immutable DSA module benchmark specification."""

    op_type: str       # OP_CONTEXT or OP_GENERATION
    batch: int
    seq_len: int       # prefill seq len (context) or KV cache len (generation)
    model_path: str    # HuggingFace model name or local path
    dtype: torch.dtype = torch.bfloat16


# ═══════════════════════════════════════════════════════════════════════
# Model config resolution — avoid HuggingFace Hub downloads
# ═══════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MODEL_CONFIGS_DIR = _PROJECT_ROOT / "model_configs"
_UPSTREAM_MODEL_CONFIGS_DIR = _PROJECT_ROOT / "src" / "aiconfigurator" / "model_configs"
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local directory path for model_name if a cached config exists."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]

    config_name = f"{model_name.replace('/', '--')}_config.json"
    for config_dir in [_LOCAL_MODEL_CONFIGS_DIR, _UPSTREAM_MODEL_CONFIGS_DIR]:
        config_file = config_dir / config_name
        if config_file.exists():
            tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
            os.symlink(config_file, os.path.join(tmp_dir, "config.json"))

            quant_file = config_dir / f"{model_name.replace('/', '--')}_hf_quant_config.json"
            if quant_file.exists():
                os.symlink(quant_file, os.path.join(tmp_dir, "hf_quant_config.json"))

            _local_config_cache[model_name] = tmp_dir
            return tmp_dir

    return model_name


# ═══════════════════════════════════════════════════════════════════════
# VllmConfig creation for NPU
# ═══════════════════════════════════════════════════════════════════════


def _create_npu_vllm_config(
    model_name: str,
    max_model_len: int,
    block_size: int,
    num_kv_cache_blocks: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """Create a VllmConfig suitable for NPU module-level benchmarking."""
    import types

    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm_ascend.ascend_config import init_ascend_config

    model_config = ModelConfig(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        max_model_len=max_model_len,
    )

    cache_config = CacheConfig(
        block_size=block_size,
        cache_dtype="auto",
    )
    cache_config.num_gpu_blocks = num_kv_cache_blocks
    cache_config.num_cpu_blocks = 0

    parallel_config = ParallelConfig(tensor_parallel_size=1)

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    device_config = DeviceConfig(device="npu")
    load_config = LoadConfig()
    compilation_config = CompilationConfig()

    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, i: 0.0, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )

    init_ascend_config(vllm_config)
    return vllm_config


# ═══════════════════════════════════════════════════════════════════════
# Attention module construction
# ═══════════════════════════════════════════════════════════════════════


def _create_attention_module(
    model_path: str,
    max_seq_len: int,
    max_batch_size: int,
    is_context: bool,
    device: str = "npu:0",
):
    """Create a DSA attention module with dummy weights on NPU.

    Uses vllm_ascend's AscendMultiHeadLatentAttention which properly creates
    indexer and all MLA submodules for DSA (Sparse Flash Attention).
    """
    from vllm.model_executor.layers.mla import MLAModules

    try:
        from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
    except ImportError as e:
        raise ImportError(
            "vllm_ascend.ops.mla module not found. "
            "Please upgrade vllm-ascend to a version that includes this module. "
            "Run: pip install --upgrade vllm-ascend"
        ) from e

    try:
        from vllm.utils.torch_utils import set_default_torch_dtype
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def set_default_torch_dtype(dtype):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                yield
            finally:
                torch.set_default_dtype(prev)

    local_model_path = _resolve_model_path(model_path)

    block_size = 64
    max_model_len = max(max_seq_len + 1, 4096)
    num_kv_cache_blocks = max(
        1 + math.ceil((max_seq_len + 1) / block_size) * max_batch_size,
        8192,
    )
    max_num_batched_tokens = (
        max(max_batch_size * max_seq_len, 131072) if is_context else max_batch_size
    )

    vllm_config = _create_npu_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_kv_cache_blocks=num_kv_cache_blocks,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    hf_config = vllm_config.model_config.hf_config
    hf_config.num_hidden_layers = 1
    num_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size

    with set_current_vllm_config(vllm_config):
        mla_modules = _create_mla_modules(
            hf_config=hf_config,
            hidden_size=hidden_size,
            device=device,
        )

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
        attn_module = AscendMultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            scale=1.0 / (hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim) ** 0.5,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            q_lora_rank=getattr(hf_config, "q_lora_rank", None),
            kv_lora_rank=hf_config.kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )

    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    with torch.no_grad():
        for name, param in attn_module.named_parameters():
            if param.is_meta:
                continue
            if param.dtype == torch.float32 and "scale" in name:
                param.data.fill_(1.0)
            else:
                param.normal_(mean=0.0, std=0.02)

    return attn_module, vllm_config


def _create_mla_modules(
    hf_config,
    hidden_size: int,
    device: str = "npu:0",
) -> "MLAModules":
    """Create MLAModules with dummy weights for DSA benchmarking."""
    from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear
    from vllm.model_executor.layers.mla import MLAModules
    from vllm.model_executor.layers.rotary_embedding import get_rope

    try:
        from vllm.model_executor.layers.mla import DeepseekV3Indexer
    except ImportError:
        DeepseekV3Indexer = None

    q_lora_rank = getattr(hf_config, "q_lora_rank", None)
    kv_lora_rank = hf_config.kv_lora_rank
    qk_nope_head_dim = hf_config.qk_nope_head_dim
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    v_head_dim = hf_config.v_head_dim
    num_heads = hf_config.num_attention_heads

    rope_parameters = {
        "rotary_dim": qk_rope_head_dim,
        "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
    }
    rotary_emb = get_rope(
        head_size=qk_rope_head_dim,
        max_position=hf_config.max_position_embeddings,
        is_neox_style=False,
        rope_parameters=rope_parameters,
        dtype=torch.bfloat16,
    )

    if q_lora_rank is None:
        q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * (qk_nope_head_dim + qk_rope_head_dim),
            bias=False,
            gather_output=False,
        )
        q_b_proj = None
        kv_a_proj_with_mqa = RowParallelLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
            input_is_parallel=True,
        )
        fused_qkv_a_proj = None
    else:
        q_proj = None
        fused_qkv_a_proj = ColumnParallelLinear(
            hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            bias=False,
            gather_output=False,
        )
        q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * (qk_nope_head_dim + qk_rope_head_dim),
            bias=False,
            gather_output=False,
        )
        kv_a_proj_with_mqa = None

    kv_b_proj = RowParallelLinear(
        kv_lora_rank,
        num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
        input_is_parallel=True,
    )
    o_proj = RowParallelLinear(
        num_heads * v_head_dim,
        hidden_size,
        bias=False,
        input_is_parallel=True,
    )

    indexer = None
    if hasattr(hf_config, "index_topk"):
        if DeepseekV3Indexer is not None:
            indexer = DeepseekV3Indexer(
                n_head=hf_config.index_n_heads,
                head_dim=hf_config.index_head_dim,
                topk_tokens=hf_config.index_topk,
                q_lora_rank=q_lora_rank if q_lora_rank else hidden_size,
                block_size=64,
            )
        else:
            indexer = _create_simple_indexer(
                n_head=hf_config.index_n_heads,
                head_dim=hf_config.index_head_dim,
                topk_tokens=hf_config.index_topk,
                q_lora_rank=q_lora_rank if q_lora_rank else hidden_size,
            )

    from vllm.model_executor.layers.norm import RMSNorm
    q_a_layernorm = RMSNorm(q_lora_rank, eps=hf_config.rms_norm_eps) if q_lora_rank else None
    kv_a_layernorm = RMSNorm(kv_lora_rank + qk_rope_head_dim, eps=hf_config.rms_norm_eps)

    return MLAModules(
        rotary_emb=rotary_emb,
        fused_qkv_a_proj=fused_qkv_a_proj,
        q_b_proj=q_b_proj,
        q_proj=q_proj,
        kv_a_proj_with_mqa=kv_a_proj_with_mqa,
        kv_a_layernorm=kv_a_layernorm,
        q_a_layernorm=q_a_layernorm,
        kv_b_proj=kv_b_proj,
        o_proj=o_proj,
        indexer=indexer,
        is_sparse=indexer is not None,
    )


class SimpleIndexer(nn.Module):
    """Simple Indexer for DSA benchmarking when DeepseekV3Indexer is unavailable."""

    def __init__(
        self,
        n_head: int,
        head_dim: int,
        topk_tokens: int,
        q_lora_rank: int,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.topk_tokens = topk_tokens
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = 1.0 / (head_dim ** 0.5)

        self.wq_b = nn.Linear(q_lora_rank, n_head * head_dim, bias=False)
        self.wk = nn.Linear(head_dim, head_dim, bias=False)
        self.weights_proj = nn.Linear(n_head * head_dim, topk_tokens, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, x):
        return x


def _create_simple_indexer(
    n_head: int,
    head_dim: int,
    topk_tokens: int,
    q_lora_rank: int,
) -> SimpleIndexer:
    """Create a simple indexer for benchmarking."""
    return SimpleIndexer(
        n_head=n_head,
        head_dim=head_dim,
        topk_tokens=topk_tokens,
        q_lora_rank=q_lora_rank,
    )


def _process_module_weights(attn_module, vllm_config) -> None:
    """Process weights after loading (creates W_UK_T, W_UV for MLA)."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    with set_current_vllm_config(vllm_config):
        mla_attn = attn_module.mla_attn
        if isinstance(mla_attn, MLAAttention) and hasattr(mla_attn, "process_weights_after_loading"):
            mla_attn.process_weights_after_loading(vllm_config.model_config.dtype)


# ═══════════════════════════════════════════════════════════════════════
# KV cache and attention metadata
# ═══════════════════════════════════════════════════════════════════════


def _create_kv_cache_and_metadata(
    vllm_config,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    device: str = "npu:0",
):
    """Create paged KV cache tuple and AscendSFAMetadata for one benchmark point.

    AscendSFAImpl.forward() expects kv_cache as a 3-tuple:
      kv_cache[0]: k_nope  [num_blocks, block_size, 1, kv_lora_rank]
      kv_cache[1]: k_pe    [num_blocks, block_size, 1, qk_rope_head_dim]
      kv_cache[2]: k_li    [num_blocks, block_size, 1, index_head_dim]  (indexer)
    All bfloat16 for GLM-5 (use_torch_npu_lightning_indexer=True, not c8).
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    hf_config = vllm_config.model_config.hf_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    index_head_dim = getattr(hf_config, "index_head_dim", 128)
    block_size = vllm_config.cache_config.block_size

    blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks = max(batch_size * blocks_per_seq, 8)

    kv_nope = torch.zeros(
        num_blocks, block_size, 1, kv_lora_rank,
        dtype=torch.bfloat16, device=device,
    )
    kv_pe = torch.zeros(
        num_blocks, block_size, 1, qk_rope_head_dim,
        dtype=torch.bfloat16, device=device,
    )
    # Indexer key cache: bfloat16 for GLM-5 (npu_lightning_indexer path)
    k_li = torch.zeros(
        num_blocks, block_size, 1, index_head_dim,
        dtype=torch.bfloat16, device=device,
    )
    kv_nope.normal_(0.0, 0.02)
    kv_pe.normal_(0.0, 0.02)
    k_li.normal_(0.0, 0.02)

    kv_cache_tuple = (kv_nope, kv_pe, k_li)

    block_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).reshape(batch_size, blocks_per_seq)

    if is_context:
        num_tokens = batch_size * seq_len
        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
        cum_query_lens = torch.arange(
            seq_len, num_tokens + 1, seq_len, dtype=torch.int32, device=device
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
        cos = torch.ones(num_tokens, 1, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        sin = torch.zeros(num_tokens, 1, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        metadata = AscendSFAMetadata(
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            attn_mask=attn_mask,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_prefills=batch_size,
        )
    else:
        num_tokens = batch_size
        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
        cum_query_lens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
        slot_mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
        cos = torch.ones(num_tokens, 1, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        sin = torch.zeros(num_tokens, 1, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        metadata = AscendSFAMetadata(
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            attn_mask=None,
            attn_state=AscendAttentionState.DecodeOnly,
            num_decodes=batch_size,
            num_decode_tokens=batch_size,
        )

    return kv_cache_tuple, metadata


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def create_dsa_module_func(
    spec: DsaModuleSpec,
    device: str = "npu:0",
) -> tuple[Callable[[], None], dict]:
    """Build and return (forward_fn, meta) for one DSA module benchmark point.

    Returns:
        forward_fn: zero-argument callable that runs one forward pass.
        meta: dict with keys num_heads, hf_config, vllm_config — needed by
              the collector to write output rows.

    Raises:
        RuntimeError: if module creation or dry run fails.
    """
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type!r}")

    is_context = spec.op_type == OP_CONTEXT

    attn_module, vllm_config = _create_attention_module(
        model_path=spec.model_path,
        max_seq_len=spec.seq_len,
        max_batch_size=spec.batch,
        is_context=is_context,
        device=device,
    )

    try:
        _process_module_weights(attn_module, vllm_config)
    except Exception:
        pass  # non-fatal; weights still usable for benchmarking

    hf_config = vllm_config.model_config.hf_config

    with set_current_vllm_config(vllm_config):
        kv_cache_tuple, attn_metadata = _create_kv_cache_and_metadata(
            vllm_config=vllm_config,
            batch_size=spec.batch,
            seq_len=spec.seq_len,
            is_context=is_context,
            device=device,
        )

    attn_layer_name = "model.layers.0.self_attn.attn"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    if attn_layer_name in forward_ctx:
        forward_ctx[attn_layer_name].kv_cache = [kv_cache_tuple]

    hidden_size = hf_config.hidden_size
    if is_context:
        num_tokens = spec.seq_len * spec.batch
        positions = (
            torch.arange(spec.seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(spec.batch, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = spec.batch
        positions = torch.full(
            (spec.batch,), spec.seq_len - 1, device=device, dtype=torch.long
        )

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    attn_metadata_dict = {attn_layer_name: attn_metadata}

    # Keep context managers alive for the lifetime of forward_fn via closure.
    _stack = ExitStack()
    _stack.enter_context(set_current_vllm_config(vllm_config))
    _stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))

    def forward_fn() -> None:
        attn_module.forward(positions, hidden_states, None)

    # Dry run
    try:
        with torch.inference_mode():
            forward_fn()
    except Exception as e:
        _stack.close()
        raise RuntimeError(
            f"DSA module dry run failed "
            f"(op={spec.op_type} b={spec.batch} s={spec.seq_len}): {e}"
        ) from e

    meta = {
        "num_heads": hf_config.num_attention_heads,
        "hf_config": hf_config,
        "vllm_config": vllm_config,
        "_stack": _stack,  # caller must call _stack.close() after benchmarking
    }
    return forward_fn, meta
