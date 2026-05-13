"""DSA/MLA module factory for Ascend NPU benchmarking — Method C, bf16 baseline.

Strategy:
  Instead of hand-building every Linear layer and faking the W8A8
  quantisation state (the previous "synthetic" approach, preserved in
  mla_module_factory.py.synthetic.bak), construct the attention module
  through vllm-ascend's own path:

    1. Build a minimal but real VllmConfig (1 hidden layer).
    2. Use vllm.model_executor.models.deepseek_v2.DeepseekV2MLAAttention
       to construct the whole module at once (projections + MLA +
       o_proj). GlmMoeDsaForCausalLM reuses DeepSeek-V3 shape config,
       so this covers GLM-5 DSA too.
    3. Let vllm-ascend's platform selector pick the attention backend
       (AscendSFABackend for use_sparse=True / use_mla=True).
    4. Build AscendSFAMetadata through the backend's metadata builder,
       feeding an AscendCommonAttentionMetadata we synthesise for the
       batch/seq point under test.
    5. Bind the KV-cache tuple into static_forward_context and run
       forward().

This version benchmarks a bf16 baseline — quant_config=None. The W8A8
variant is a follow-up once the bf16 path is green end-to-end.
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


def _ensure_c_ascend_loaded() -> None:
    """Force-load vllm_ascend_C.so so torch.ops._C_ascend.* work."""
    import glob

    try:
        import vllm_ascend  # type: ignore
    except ImportError:
        return

    root = os.path.dirname(vllm_ascend.__file__)
    for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so)
        except Exception as e:
            print(f"[WARN] failed to load {so}: {e}")


def _ensure_npu_compile_opts() -> None:
    """Mirror vllm_ascend/worker/worker.py:_init_device() init sequence.

    Without these, AclSetCompileopt / npu_format_cast(29) fail with
    error 500001 when later op paths try to use NZ layout.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return

    # allow_internal_format must be set BEFORE set_device; later toggles
    # are silently ignored once ACL context is locked in.
    try:
        torch.npu.config.allow_internal_format = True
    except Exception as e:
        print(f"[WARN] set allow_internal_format failed: {type(e).__name__}: {e}")

    try:
        torch.npu.set_device(0)
    except Exception as e:
        print(f"[WARN] torch.npu.set_device failed: {type(e).__name__}: {e}")

    try:
        import torch_npu._inductor  # noqa: F401
    except Exception as e:
        print(f"[WARN] import torch_npu._inductor failed: {type(e).__name__}: {e}")

    try:
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception as e:
        print(f"[WARN] set_compile_mode failed: {type(e).__name__}: {e}")

    try:
        import gc
        _ = torch.zeros(1, device="npu:0") + 1
        del _
        gc.collect()
        torch.npu.empty_cache()
    except Exception as e:
        print(f"[WARN] NPU warmup failed: {type(e).__name__}: {e}")


_ensure_c_ascend_loaded()
_ensure_npu_compile_opts()

from vllm.config import set_current_vllm_config

# Register GLM-5 DSA config to DeepSeek-V3 so AutoConfig can resolve it.
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
    """Immutable module benchmark specification."""

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

    # If model_name is already a valid directory with config.json, use as-is.
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
        _local_config_cache[model_name] = model_name
        return model_name

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
# VllmConfig for NPU (reused from the synthetic version, unchanged)
# ═══════════════════════════════════════════════════════════════════════


def _create_npu_vllm_config(
    model_name: str,
    max_model_len: int,
    block_size: int,
    num_kv_cache_blocks: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """Create a VllmConfig suitable for NPU module-level benchmarking.

    Pins num_hidden_layers=1 so the model is cheap to instantiate, then
    lets vllm-ascend's Ascend platform wire up the rest of the config.
    """
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

    # Pin 1 hidden layer in the HF config so DeepseekV2MLAAttention
    # construction is cheap.
    model_config.hf_config.num_hidden_layers = 1

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
    # bf16 baseline — no quantization. W8A8 follow-up can replace this.
    vllm_config.quant_config = None
    return vllm_config


# ═══════════════════════════════════════════════════════════════════════
# Attention module via DeepseekV2MLAAttention (vllm upstream)
# ═══════════════════════════════════════════════════════════════════════


def _set_default_dtype_ctx(dtype):
    """set_default_torch_dtype context manager, with a fallback for older vllm."""
    try:
        from vllm.utils.torch_utils import set_default_torch_dtype
        return set_default_torch_dtype(dtype)
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def _fallback(dtype):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                yield
            finally:
                torch.set_default_dtype(prev)

        return _fallback(dtype)


def _build_attention_module(
    model_path: str,
    max_seq_len: int,
    max_batch_size: int,
    is_context: bool,
    device: str,
):
    """Build a DeepseekV2MLAAttention module with real vllm-ascend wiring.

    Steps mirror aiconfigurator GPU's collect_mla_module.py:
      1. Resolve the model_path to a local config directory.
      2. Build a minimal VllmConfig via _create_npu_vllm_config.
      3. Instantiate DeepseekV2MLAAttention under set_current_vllm_config
         + bf16 default dtype.
      4. Move to NPU; fill parameters with random bf16 values (real
         post-loading layout emerges from process_weights_after_loading).
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

    local_model_path = _resolve_model_path(model_path)

    # Scale num_kv_cache_blocks so every query can be placed and the
    # indexer/sparse kernels have enough KV slots.
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

    # vllm-ascend registers AscendMultiHeadLatentAttention / AscendRMSNorm /
    # AscendColumnParallelLinear etc. via CustomOp.register_oot(). Without
    # that registration, vllm's `MultiHeadLatentAttentionWrapper(...)`
    # returns the generic vllm-upstream wrapper which doesn't forward
    # `rotary_emb` (and several other MLA-module kwargs) to AscendSFAImpl,
    # making its __init__ raise KeyError('rotary_emb'). The registration
    # normally happens inside vllm_ascend.worker.worker.Worker, which
    # AIConfigurator bypasses -- so trigger it manually here.
    try:
        from vllm_ascend.utils import register_ascend_customop
        register_ascend_customop(vllm_config)
    except ImportError as e:
        print(f"[WARN] register_ascend_customop unavailable: {e}")

    hf_config = vllm_config.model_config.hf_config
    num_heads = hf_config.num_attention_heads

    # DSA indexer (GlmMoeDsa / DeepSeek-V3.2) needs a pre-allocated
    # topk_indices_buffer sized to max_num_batched_tokens × index_topk.
    topk_indices_buffer = None
    if hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(
            max_tokens,
            hf_config.index_topk,
            dtype=torch.int32,
            device=device,
        )

    with set_current_vllm_config(vllm_config), _set_default_dtype_ctx(torch.bfloat16):
        attn_module = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_config,
            hidden_size=hf_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            q_lora_rank=getattr(hf_config, "q_lora_rank", None),
            kv_lora_rank=hf_config.kv_lora_rank,
            max_position_embeddings=hf_config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

    # Move to NPU (meta params need to_empty).
    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # Random init matching aiconfigurator GPU conventions:
    #   fp8 weights → 0, fp32 scale params → 1.0, everything else → N(0, 0.02).
    with torch.no_grad():
        for name, param in attn_module.named_parameters():
            if param.is_meta:
                continue
            if param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                param.data.zero_()
            elif param.dtype == torch.float32 and "scale" in name:
                param.data.fill_(1.0)
            else:
                param.normal_(mean=0.0, std=0.02)

    return attn_module, vllm_config


def _process_module_weights(attn_module, vllm_config) -> None:
    """Mirror vllm's model loader post-load hooks.

    Two passes, same as aiconfigurator GPU version:
      1. quant_method.process_weights_after_loading on every layer that
         has one (runs FP8 / W8A8 packing).
      2. MLAAttention.process_weights_after_loading(dtype) on the MLA
         attention leaf (materialises W_UK_T / W_UV and MLAPO fused
         weights when enable_mlapo=True).
    """
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    except ImportError:
        MLAAttention = None  # type: ignore

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            qm = getattr(module, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                try:
                    qm.process_weights_after_loading(module)
                except Exception as e:
                    print(
                        f"[WARN] quant_method.process_weights_after_loading failed "
                        f"on {type(module).__name__}: {type(e).__name__}: {e}"
                    )

        if MLAAttention is not None:
            for _, module in attn_module.named_modules():
                if isinstance(module, MLAAttention) and hasattr(
                    module, "process_weights_after_loading"
                ):
                    module.process_weights_after_loading(vllm_config.model_config.dtype)


# ═══════════════════════════════════════════════════════════════════════
# KV cache + metadata via AscendSFABackend builder
# ═══════════════════════════════════════════════════════════════════════


def _create_common_attn_metadata(
    batch_size: int,
    seq_len: int,
    is_context: bool,
    block_size: int,
    num_blocks: int,
    device: str,
):
    """Synthesise AscendCommonAttentionMetadata for one benchmark point.

    Ascend's builder needs more fields than vanilla vllm's
    CommonAttentionMetadata — notably positions / actual_seq_lengths_q /
    attn_state / num_input_tokens — so we build the Ascend subclass
    directly rather than relying on vllm's test helper.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    if is_context:
        query_lens = [seq_len] * batch_size
        seq_lens_list = [seq_len] * batch_size
        num_input_tokens = batch_size * seq_len
        attn_state = AscendAttentionState.PrefillNoCache
    else:
        query_lens = [1] * batch_size
        seq_lens_list = [seq_len] * batch_size
        num_input_tokens = batch_size
        attn_state = AscendAttentionState.DecodeOnly

    query_start_loc_cpu = torch.tensor(
        [0] + [sum(query_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
    )
    query_start_loc = query_start_loc_cpu.to(device)

    seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
    seq_lens_t = seq_lens_cpu.to(device)

    num_computed_tokens_cpu = torch.zeros(batch_size, dtype=torch.int32)

    # block_table: one row per request, enough columns to cover seq_len.
    blocks_per_row = max(math.ceil(seq_len / block_size), 1)
    bt = torch.arange(batch_size * blocks_per_row, dtype=torch.int32, device=device)
    bt = (bt % num_blocks).reshape(batch_size, blocks_per_row)

    # Pad up to max_blocks so the builder can safely slice.
    max_blocks = math.ceil(num_blocks * block_size / block_size)
    if blocks_per_row < max_blocks:
        pad = (max_blocks - blocks_per_row)
        bt = torch.cat(
            [bt, torch.zeros(batch_size, pad, dtype=torch.int32, device=device)], dim=1
        )

    # slot_mapping: for each input token, where it lives in KV cache.
    slot_mapping = torch.arange(num_input_tokens, dtype=torch.int32, device=device)

    # positions: one int64 per input token.
    if is_context:
        positions = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        positions = torch.full(
            (batch_size,), seq_len - 1, dtype=torch.long, device=device
        )

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_t,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=num_input_tokens,
        max_query_len=max(query_lens),
        block_table_tensor=bt,
        slot_mapping=slot_mapping,
        causal=True,
        actual_seq_lengths_q=list(query_start_loc_cpu[1:].tolist()),
        positions=positions,
        attn_state=attn_state,
        graph_pad_size=-1,
        num_input_tokens=num_input_tokens,
        max_seq_len=max(seq_lens_list),
        decode_token_per_req=1,
    )


def _create_kv_cache_and_metadata(
    vllm_config,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    device: str,
):
    """Create the 3-tensor KV cache tuple and AscendSFAMetadata.

    vllm-ascend's AscendSFAImpl expects kv_cache as a 3-tuple:
      [0] k_nope  (num_blocks, block_size, 1, kv_lora_rank)
      [1] k_pe    (num_blocks, block_size, 1, qk_rope_head_dim)
      [2] k_li    (num_blocks, block_size, 1, index_head_dim)  DSA only
    Builder is obtained through AscendSFABackend.get_builder_cls().
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend

    hf_config = vllm_config.model_config.hf_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim
    index_head_dim = getattr(hf_config, "index_head_dim", 128)
    index_topk = getattr(hf_config, "index_topk", 2048)
    block_size = vllm_config.cache_config.block_size

    # Scale num_blocks to cover both this request's KV and the sparse
    # kernel's index_topk reach.
    min_blocks = math.ceil(max(seq_len, index_topk) / block_size)
    num_blocks = max(batch_size * min_blocks, 8)

    kv_nope = torch.randn(
        num_blocks, block_size, 1, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    kv_pe = torch.randn(
        num_blocks, block_size, 1, qk_rope_head_dim, dtype=torch.bfloat16, device=device
    )
    k_li = torch.randn(
        num_blocks, block_size, 1, index_head_dim, dtype=torch.bfloat16, device=device
    )
    kv_cache_tuple = (kv_nope, kv_pe, k_li)

    common_meta = _create_common_attn_metadata(
        batch_size=batch_size,
        seq_len=seq_len,
        is_context=is_context,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
    )

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=torch.bfloat16,
        sliding_window=None,
        cache_dtype_str=None,
    )

    layer_names = ["model.layers.0.self_attn.attn"]
    builder_cls = AscendSFABackend.get_builder_cls()
    builder = builder_cls(
        kv_cache_spec, layer_names, vllm_config, torch.device(device)
    )
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_meta,
    )

    return kv_cache_tuple, attn_metadata


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def create_dsa_module_func(
    spec: DsaModuleSpec,
    device: str = "npu:0",
) -> tuple[Callable[[], None], dict]:
    """Build and return (forward_fn, meta) for one benchmark point.

    Returns:
        forward_fn: zero-argument callable that runs one forward pass.
        meta: dict with hf_config / vllm_config / _stack (ExitStack the
              caller must close after benchmarking).

    Raises:
        RuntimeError: if module construction or dry run fails.
    """
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type!r}")
    is_context = spec.op_type == OP_CONTEXT

    # 1. Attention module
    attn_module, vllm_config = _build_attention_module(
        model_path=spec.model_path,
        max_seq_len=spec.seq_len,
        max_batch_size=spec.batch,
        is_context=is_context,
        device=device,
    )

    # 2. Post-load hooks (FP8 packing, W_UK_T materialisation, etc).
    _process_module_weights(attn_module, vllm_config)

    # 3. KV cache + attn_metadata
    with set_current_vllm_config(vllm_config):
        kv_cache_tuple, attn_metadata = _create_kv_cache_and_metadata(
            vllm_config=vllm_config,
            batch_size=spec.batch,
            seq_len=spec.seq_len,
            is_context=is_context,
            device=device,
        )

    # 4. Bind KV cache to the attn layer so its forward() can read it.
    attn_layer_name = "model.layers.0.self_attn.attn"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    if attn_layer_name in forward_ctx:
        forward_ctx[attn_layer_name].kv_cache = [kv_cache_tuple]

    # 5. Hidden states + positions inputs.
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    if is_context:
        num_tokens = spec.batch * spec.seq_len
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
            (spec.batch,), spec.seq_len - 1, dtype=torch.long, device=device
        )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    # 6. Forward context — vllm-ascend ships set_ascend_forward_context,
    # fall back to vllm's set_forward_context if that import fails.
    stack = ExitStack()
    stack.enter_context(set_current_vllm_config(vllm_config))

    attn_metadata_dict = {attn_layer_name: attn_metadata}
    try:
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import set_weight_prefetch_method
        from vllm_ascend.ascend_config import WeightPrefetchConfig

        set_weight_prefetch_method(WeightPrefetchConfig({}))

        num_tokens_across_dp = torch.tensor(
            [num_tokens], dtype=torch.int64, device=device
        )
        stack.enter_context(
            set_ascend_forward_context(
                attn_metadata_dict,
                vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            )
        )
    except ImportError:
        from vllm.forward_context import set_forward_context
        stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))

    def forward_fn() -> None:
        attn_module.forward(positions, hidden_states, None)

    # 7. Dry run — surface failures here instead of during benchmarking.
    try:
        with torch.inference_mode():
            forward_fn()
    except Exception as e:
        stack.close()
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Module dry run failed "
            f"(op={spec.op_type} b={spec.batch} s={spec.seq_len}): {e}"
        ) from e

    meta = {
        "num_heads": vllm_config.model_config.hf_config.num_attention_heads,
        "hf_config": vllm_config.model_config.hf_config,
        "vllm_config": vllm_config,
        "_stack": stack,  # caller must call _stack.close()
    }
    return forward_fn, meta
