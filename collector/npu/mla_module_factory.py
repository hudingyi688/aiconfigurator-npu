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


def _ensure_c_ascend_loaded() -> None:
    """Force-load vllm_ascend_C.so — see collect_mla_module.py for rationale."""
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
    """Apply NPU compile options that vllm-ascend normally sets inside
    model_runner_v1 / worker. AIConfigurator builds vllm_config manually
    and skips those paths; without set_device + allow_internal_format +
    set_compile_mode, MLAPO's process_weights_after_loading hits
    'AclSetCompileopt ... ACL_PRECISION_MODE error 500001' when it calls
    npu_format_cast(wd_qkv, 29) for the MLAPO NZ layout.

    Mirrors vllm_ascend/worker/worker.py:_init_device() ordering.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return

    # IMPORTANT: allow_internal_format must be toggled BEFORE set_device
    # or ACL context locks internal_format=False and later changes are
    # silently ignored (see the "Cannot create tensor with internal
    # format" warning). Mirrors model_runner_v1.py:156 which sets this
    # at module import time, well before any device binding.
    try:
        torch.npu.config.allow_internal_format = True
    except Exception as e:
        print(f"[WARN] set allow_internal_format failed: {type(e).__name__}: {e}")

    # Bind NPU device.
    try:
        torch.npu.set_device(0)
    except Exception as e:
        print(f"[WARN] torch.npu.set_device failed: {type(e).__name__}: {e}")

    # Import torch_npu._inductor (worker.py does this when Triton is
    # available). Appears to finish ACL compile-opt lazy init so later
    # AclSetCompileopt calls don't 500001.
    try:
        import torch_npu._inductor  # noqa: F401
    except Exception as e:
        print(f"[WARN] import torch_npu._inductor failed: {type(e).__name__}: {e}")

    # Mirror vllm_ascend/compilation/compiler_interface.py:81
    try:
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception as e:
        print(f"[WARN] set_compile_mode failed: {type(e).__name__}: {e}")

    # Trigger a tiny NPU op and empty_cache to force ACL lazy-init to
    # fully complete (what MemorySnapshot() does implicitly in worker).
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
from vllm.forward_context import set_forward_context

os.environ.setdefault("VLLM_ASCEND_ENABLE_MLAPO", "1")


def _setup_w8a8_quant_method(layer: nn.Module, input_size: int, output_size: int, dtype: torch.dtype = torch.bfloat16):
    """Setup W8A8 static quant attributes on a linear layer.

    Mirrors the post-loading state produced by
    AscendW8A8LinearMethod.process_weights_after_loading so that
    apply() → torch.ops.vllm.quantize + torch_npu.npu_quant_matmul work.
    """
    quant_method_cls = None
    try:
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
        quant_method_cls = AscendW8A8LinearMethod
    except ImportError:
        print(f"[WARNING] AscendW8A8LinearMethod not found, MLAPO may not work")

    if quant_method_cls is not None:
        qm = quant_method_cls()
        qm.quant_method = qm
        layer.quant_method = qm

    # weight: int8, stored as (input_size, output_size) after transpose in
    # process_weights_after_loading. We synthesize it directly in that layout.
    weight_float = layer.weight.data.float()
    weight_scale = weight_float.abs().max(dim=1).values.clamp(min=1e-5)   # (output_size,)
    weight_int8 = (weight_float / weight_scale.unsqueeze(1)).round().clamp(-127, 127)
    weight_int8 = weight_int8.T.contiguous().to(torch.int8)               # (input_size, output_size)
    layer.weight.data = weight_int8

    # per-channel weight params (flattened to 1-D as post-loading does)
    layer.register_buffer("weight_scale", weight_scale.to(dtype))          # bf16, (output_size,)
    layer.register_buffer("weight_offset", torch.zeros(output_size, dtype=dtype))

    # per-tensor activation params
    layer.register_buffer("input_scale", torch.ones(1, dtype=dtype))       # bf16, (1,)
    layer.register_buffer("input_offset", torch.zeros(1, dtype=torch.int8))  # int8, (1,)

    # aclnn_* are per-input-channel, shape = (input_size,). dtype = activation dtype (bf16).
    aclnn_scale = layer.input_scale.data.repeat(input_size).to(dtype)      # (input_size,)
    layer.aclnn_input_scale = nn.Parameter(aclnn_scale, requires_grad=False)
    layer.aclnn_input_scale_reciprocal = nn.Parameter(
        (1.0 / aclnn_scale).contiguous(), requires_grad=False
    )
    layer.aclnn_input_offset = nn.Parameter(
        layer.input_offset.data.repeat(input_size).to(dtype), requires_grad=False
    )

    # deq_scale: bf16 activation path uses float32; fp16 path uses int64 packed scale.
    deq_scale_dtype = torch.float32 if dtype == torch.bfloat16 else torch.int64
    deq_scale_val = (layer.input_scale.data.to(torch.float32)
                     * layer.weight_scale.data.to(torch.float32))           # (output_size,)
    layer.deq_scale = nn.Parameter(deq_scale_val.to(deq_scale_dtype), requires_grad=False)

    # quant_bias: int32, (output_size,)
    layer.quant_bias = nn.Parameter(
        torch.zeros(output_size, dtype=torch.int32), requires_grad=False
    )

    # npu_quant_matmul reads layer.params_dtype as output dtype
    layer.params_dtype = dtype


class MockW8A8Linear(nn.Module):
    """Mock W8A8 quantized Linear layer for MLAPO benchmarking.
    
    Simulates the structure of AscendW8A8DynamicLinearMethod layers
    to enable MLAPO fused path, avoiding npu_kv_rmsnorm_rope_cache
    which only supports last_dim=512 or 192 (GLM-5 uses 576).
    
    Weight is stored in transposed format (input_size, output_size)
    as expected by _process_weights_for_fused_mlapo after
    AscendW8A8DynamicLinearMethod.process_weights_after_loading.
    """

    def __init__(self, input_size: int, output_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size

        weight_float = torch.zeros(output_size, input_size, dtype=torch.float32)
        weight_float.uniform_(-1.0, 1.0)
        weight_scale = weight_float.abs().max(dim=1).values.clamp(min=1e-5)
        # int8 weight in (input_size, output_size) layout (post-loading shape)
        weight_data = (weight_float / weight_scale.unsqueeze(1)).round().clamp(-127, 127)
        weight_data = weight_data.T.contiguous().to(torch.int8)
        self.register_buffer("weight", weight_data)

        self.register_buffer("weight_scale", weight_scale.to(dtype))
        self.register_buffer("weight_offset", torch.zeros(output_size, dtype=dtype))

        self.register_buffer("input_scale", torch.ones(1, dtype=dtype))
        self.register_buffer("input_offset", torch.zeros(1, dtype=torch.int8))

        aclnn_scale = self.input_scale.data.repeat(input_size).to(dtype)
        self.aclnn_input_scale = nn.Parameter(aclnn_scale, requires_grad=False)
        self.aclnn_input_scale_reciprocal = nn.Parameter(
            (1.0 / aclnn_scale).contiguous(), requires_grad=False
        )
        self.aclnn_input_offset = nn.Parameter(
            self.input_offset.data.repeat(input_size).to(dtype), requires_grad=False
        )

        deq_scale_dtype = torch.float32 if dtype == torch.bfloat16 else torch.int64
        deq_scale_val = (self.input_scale.data.to(torch.float32)
                         * self.weight_scale.data.to(torch.float32))
        self.deq_scale = nn.Parameter(deq_scale_val.to(deq_scale_dtype), requires_grad=False)
        self.quant_bias = nn.Parameter(
            torch.zeros(output_size, dtype=torch.int32), requires_grad=False
        )

        self.params_dtype = dtype

        self.quant_config = None
        qm = self._get_quant_method()
        if qm is not None:
            qm.quant_method = qm
        self.quant_method = qm

    def _get_quant_method(self):
        try:
            from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
            return AscendW8A8LinearMethod()
        except ImportError:
            return None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self.quant_method is not None:
            try:
                return self.quant_method.apply(self, x), None
            except Exception:
                pass
        
        weight_dequant = self.weight.float() * self.weight_scale.float().unsqueeze(1) + self.weight_offset.float().unsqueeze(1)
        output = torch.matmul(x, weight_dequant.T.to(x.dtype))
        return output, None

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
        
        fused_qkv_a_proj_output_size = q_lora_rank + kv_lora_rank + qk_rope_head_dim
        fused_qkv_a_proj = ColumnParallelLinear(
            hidden_size,
            fused_qkv_a_proj_output_size,
            bias=False,
            gather_output=False,
        )
        
        q_b_proj_output_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            q_b_proj_output_size,
            bias=False,
            gather_output=False,
        )
        
        _setup_w8a8_quant_method(fused_qkv_a_proj, hidden_size, fused_qkv_a_proj_output_size)
        _setup_w8a8_quant_method(q_b_proj, q_lora_rank, q_b_proj_output_size)
        
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
                hidden_size=hidden_size,
            )
        indexer = indexer.to(torch.device(device))
        
        if hasattr(indexer, "k_norm"):
            with torch.no_grad():
                indexer.k_norm.weight.data = indexer.k_norm.weight.data.to(torch.bfloat16)
                if indexer.k_norm.bias is not None:
                    indexer.k_norm.bias.data = indexer.k_norm.bias.data.to(torch.bfloat16)

    from vllm.model_executor.layers.layernorm import RMSNorm
    
    with torch.no_grad():
        q_a_layernorm = RMSNorm(q_lora_rank, eps=hf_config.rms_norm_eps) if q_lora_rank else None
        kv_a_layernorm = RMSNorm(kv_lora_rank + qk_rope_head_dim, eps=hf_config.rms_norm_eps)
        
        if q_a_layernorm is not None:
            q_a_layernorm.weight.data = q_a_layernorm.weight.data.to(torch.bfloat16)
            q_a_layernorm.bias = nn.Parameter(torch.zeros(q_lora_rank, dtype=torch.bfloat16))
        kv_a_layernorm.weight.data = kv_a_layernorm.weight.data.to(torch.bfloat16)

    target_device = torch.device(device)
    
    if q_proj is not None:
        q_proj = q_proj.to(target_device)
    if q_b_proj is not None:
        q_b_proj = q_b_proj.to(target_device)
    if fused_qkv_a_proj is not None:
        fused_qkv_a_proj = fused_qkv_a_proj.to(target_device)
    if kv_a_proj_with_mqa is not None:
        kv_a_proj_with_mqa = kv_a_proj_with_mqa.to(target_device)
    kv_b_proj = kv_b_proj.to(target_device)
    o_proj = o_proj.to(target_device)
    if q_a_layernorm is not None:
        q_a_layernorm = q_a_layernorm.to(target_device)
    kv_a_layernorm = kv_a_layernorm.to(target_device)

    topk_indices_buffer = None
    if indexer is not None:
        topk_indices_buffer = torch.empty(
            4096,
            hf_config.index_topk,
            dtype=torch.int32,
            device=device,
        )

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
        topk_indices_buffer=topk_indices_buffer,
    )


class _TupleLinear(nn.Module):
    """nn.Linear wrapper whose __call__ returns (output, None).

    vllm linear layers (ColumnParallelLinear/RowParallelLinear) return
    an (output, bias) 2-tuple, and sfa_v1.py uses
      q_li, _ = self.wq_b(q_c)
      weights, _ = self.weights_proj(x)
    Plain nn.Linear returns a single tensor, which breaks 2-tuple
    unpacking with ValueError. Use this wrapper for any indexer Linear
    that sfa_v1.py unpacks.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        return self._linear(x), None


class SimpleIndexer(nn.Module):
    """Simple Indexer for DSA benchmarking when DeepseekV3Indexer is unavailable."""

    def __init__(
        self,
        n_head: int,
        head_dim: int,
        topk_tokens: int,
        q_lora_rank: int,
        hidden_size: int,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.topk_tokens = topk_tokens
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = 1.0 / (head_dim ** 0.5)

        # vllm linear layers return (output, bias) 2-tuples; sfa_v1.py does
        #   q_li, _ = self.wq_b(q_c)
        #   weights, _ = self.weights_proj(x)
        # so we wrap nn.Linear in a thin module that mimics that interface.
        self.wq_b = _TupleLinear(q_lora_rank, n_head * head_dim)
        self._wk_linear = nn.Linear(hidden_size, head_dim, bias=False)
        # weights_proj takes hidden_states (not wq_b output) as input, and
        # produces per-head scalar weights used to combine q_li per head.
        # See vllm_ascend/attention/sfa_v1.py:956 where
        #   weights, _ = self.weights_proj(x)
        # and x is hidden_states with last dim = hidden_size.
        # Output dim is n_head (one weight per indexer head), NOT topk_tokens.
        self.weights_proj = _TupleLinear(hidden_size, n_head)
        self.k_norm = nn.LayerNorm(head_dim)
        
        with torch.no_grad():
            self.k_norm.weight.data = self.k_norm.weight.data.to(torch.bfloat16)
            if self.k_norm.bias is not None:
                self.k_norm.bias.data = self.k_norm.bias.data.to(torch.bfloat16)

    def wk(self, x):
        """Return tuple (output, None) to match vllm linear layer interface."""
        return self._wk_linear(x), None

    def forward(self, x):
        return x


def _create_simple_indexer(
    n_head: int,
    head_dim: int,
    topk_tokens: int,
    q_lora_rank: int,
    hidden_size: int,
) -> SimpleIndexer:
    """Create a simple indexer for benchmarking."""
    return SimpleIndexer(
        n_head=n_head,
        head_dim=head_dim,
        topk_tokens=topk_tokens,
        q_lora_rank=q_lora_rank,
        hidden_size=hidden_size,
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
    # sparse_count is the number of KV slots lightning_indexer must pick
    # from the cache; the kernel will index into kv_cache[0..num_blocks*bs-1]
    # assuming that many valid slots exist. Without this reserve the kernel
    # reads out of bounds and segfaults when seq_len is small.
    index_topk = getattr(hf_config, "index_topk", 2048)
    block_size = vllm_config.cache_config.block_size

    blocks_per_seq = math.ceil(seq_len / block_size)
    # num_blocks must satisfy:
    #   1) cover every query token in block_table (batch_size * blocks_per_seq)
    #   2) hold at least index_topk slots so sparse_flash_attention's gather
    #      stays in-bounds (num_blocks * block_size >= index_topk)
    #   3) a small floor so kernels with static tiling don't underflow
    min_blocks_for_topk = math.ceil(index_topk / block_size)
    num_blocks = max(batch_size * blocks_per_seq, min_blocks_for_topk, 8)

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

    # block_table must advertise enough blocks so downstream sparse kernels
    # can gather up to index_topk tokens (= ceil(index_topk / block_size)
    # blocks) per query. We tile num_blocks across each row.
    blocks_per_row = max(blocks_per_seq, min_blocks_for_topk)
    # Wrap-around so every entry is in [0, num_blocks).
    block_table = (
        torch.arange(batch_size * blocks_per_row, dtype=torch.int32, device=device)
        % num_blocks
    ).reshape(batch_size, blocks_per_row)

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
    except Exception as e:
        # Log full traceback so we can see which NPU op inside
        # process_weights_after_loading triggers ACL errors.
        import traceback
        print(
            f"[WARN] _process_module_weights failed: {type(e).__name__}: {e}. "
            f"Disabling MLAPO for this module (enable_mlapo=False)."
        )
        traceback.print_exc()
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        for _, sub in attn_module.named_modules():
            if isinstance(sub, MLAAttention) and hasattr(sub, "impl"):
                if hasattr(sub.impl, "enable_mlapo"):
                    sub.impl.enable_mlapo = False

    # Diagnostic: report the MLAPO decision made by process_weights_after_loading
    # so we can see why forward takes the non-MLAPO branch (which then hits
    # npu_kv_rmsnorm_rope_cache's last_dim in {512,192} limitation on GLM-5).
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod
        for name, sub in attn_module.named_modules():
            if isinstance(sub, MLAAttention) and hasattr(sub, "impl"):
                impl = sub.impl
                fqa = getattr(impl, "fused_qkv_a_proj", None)
                qm_wrap = getattr(fqa, "quant_method", None) if fqa is not None else None
                qm_inner = getattr(qm_wrap, "quant_method", None)
                print(
                    f"[MLAPO DIAG {name}] "
                    f"enable_mlapo={getattr(impl, 'enable_mlapo', None)} "
                    f"enable_dsa_cp={getattr(impl, 'enable_dsa_cp', None)} "
                    f"has_wd_qkv={hasattr(impl, 'wd_qkv')} "
                    f"fused_qkv_a_proj={'set' if fqa is not None else 'None'} "
                    f"qm_wrap_type={type(qm_wrap).__name__ if qm_wrap is not None else 'None'} "
                    f"qm_inner_type={type(qm_inner).__name__ if qm_inner is not None else 'None'} "
                    f"isinstance_W8A8={isinstance(qm_inner, AscendW8A8LinearMethod)}"
                )
    except Exception as e:
        print(f"[MLAPO DIAG] introspection failed: {type(e).__name__}: {e}")

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

    num_tokens_across_dp = torch.tensor([num_tokens], dtype=torch.int64, device=device)

    # Keep context managers alive for the lifetime of forward_fn via closure.
    _stack = ExitStack()
    _stack.enter_context(set_current_vllm_config(vllm_config))

    attn_metadata_dict = {attn_layer_name: attn_metadata}

    try:
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import set_weight_prefetch_method
        from vllm_ascend.ascend_config import WeightPrefetchConfig
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        
        try:
            init_device_properties_triton()
        except Exception:
            pass
        
        weight_prefetch_config = WeightPrefetchConfig({})
        set_weight_prefetch_method(weight_prefetch_config)
        
        _stack.enter_context(set_ascend_forward_context(
            attn_metadata_dict,
            vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        ))
    except ImportError:
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
