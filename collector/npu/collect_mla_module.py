"""DSA Module Collector for Ascend NPU — Module-level (Method C).

Profiles the complete DeepseekV2MLAAttention module forward pass
(projections + MLA attention + output), not just the bare attention kernel.

Calling chain (GLM-5 on NPU):
  collect_mla_module.py
    → DeepseekV2MLAAttention.forward()
      → fused_qkv_a_proj → q_a_layernorm → q_b_proj → Q
      → kv_a_proj → kv_a_layernorm → kv_c + k_pe
      → AscendSFAImpl.forward()   ← DSA backend (use_sparse=True)
        → indexer_select_pre_process / post_process
        → npu_sparse_flash_attention
      → o_proj → output

Output: dsa_context_module_perf.txt / dsa_generation_module_perf.txt
        (aiconfigurator DSA module format, ready for config search)

Usage:
    python collector/npu/collect_mla_module.py --mode context
    python collector/npu/collect_mla_module.py --mode generation
    python collector/npu/collect_mla_module.py --mode context --quick --batch-size 4 --seq-len 2048
"""

import argparse
import gc
import math
import os
import tempfile
import traceback
from contextlib import ExitStack
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context

# Patch config registry so AutoConfig resolves glm_moe_dsa → DeepseekV3Config.
# GlmMoeDsaForCausalLM inherits DeepseekV2ForCausalLM; the config layout is
# identical to DeepSeek-V3, so reusing DeepseekV3Config is safe.
try:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    if "glm_moe_dsa" not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"
except Exception:
    pass

from bench_engine import BenchResult, benchmark_npu

# ═══════════════════════════════════════════════════════════════════════
# Model config resolution — avoid HuggingFace Hub downloads
# ═══════════════════════════════════════════════════════════════════════

# Pre-cached HF configs live in aiconfigurator's model_configs/ directory.
_MODEL_CONFIGS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src" / "aiconfigurator" / "model_configs"
)
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local directory path for model_name if a cached config exists."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]

    config_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_config.json"
    if not config_file.exists():
        return model_name

    tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
    os.symlink(config_file, os.path.join(tmp_dir, "config.json"))

    quant_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_hf_quant_config.json"
    if quant_file.exists():
        os.symlink(quant_file, os.path.join(tmp_dir, "hf_quant_config.json"))

    _local_config_cache[model_name] = tmp_dir
    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Supported models
# ═══════════════════════════════════════════════════════════════════════

SUPPORTED_MODELS: dict[str, str] = {
    "zai-org/GLM-5": "dsa",
}

# ═══════════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════════

_CONTEXT_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
_CONTEXT_SEQ_LIST = [
    1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048,
    3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768,
]
_GENERATION_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_GENERATION_SEQ_LIST = [
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072,
]


def get_context_test_cases() -> list[tuple]:
    """Return (seq_len, batch_size, num_heads, model_path) tuples for context phase."""
    cases = []
    for model_path in SUPPORTED_MODELS:
        for b in _CONTEXT_BATCH_LIST:
            for s in _CONTEXT_SEQ_LIST:
                if b * s > 131072:
                    continue
                cases.append((s, b, model_path))
    return cases


def get_generation_test_cases() -> list[tuple]:
    """Return (kv_cache_len, batch_size, model_path) tuples for generation phase."""
    cases = []
    for model_path in SUPPORTED_MODELS:
        for b in _GENERATION_BATCH_LIST:
            for s in _GENERATION_SEQ_LIST:
                if b * s > 1024 * 4096 * 8:
                    continue
                cases.append((s, b, model_path))
    return cases


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
        swap_space=0,
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

    # Add mock methods required by some attention backends.
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
    """Create a DeepseekV2MLAAttention module with dummy weights on NPU."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

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

    # DSA topk_indices_buffer
    topk_indices_buffer = None
    if hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(
            max_tokens,
            hf_config.index_topk,
            dtype=torch.int32,
            device=device,
        )

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16):
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
            quant_config=None,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=topk_indices_buffer,
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


def _process_module_weights(attn_module, vllm_config) -> None:
    """Process weights after loading (creates W_UK_T, W_UV for MLA)."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            if isinstance(module, MLAAttention) and hasattr(
                module, "process_weights_after_loading"
            ):
                module.process_weights_after_loading(
                    vllm_config.model_config.dtype
                )


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

    # Three separate paged KV cache tensors for SFA
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
        # cum_query_lens: cumulative token counts per request [seq_len, 2*seq_len, ...]
        cum_query_lens = torch.arange(
            seq_len, num_tokens + 1, seq_len, dtype=torch.int32, device=device
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
        # sin/cos: [num_tokens, 1, 1, qk_rope_head_dim]
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
        # cum_query_lens for decode: [1, 2, ..., batch_size]
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
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


def run_dsa_module(
    seq_len: int,
    batch_size: int,
    model_path: str,
    mode: str,
    output_dir: Path,
    device: str = "npu:0",
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> float | None:
    """Benchmark one (seq_len, batch_size) point for DSA module on NPU."""
    is_context = mode == "context"
    phase = "context" if is_context else "generation"
    hf_config_ref = None

    print(
        f"\n[DSA module] {phase} b={batch_size}, s={seq_len}, model={model_path}"
    )

    with ExitStack() as stack:
        try:
            # 1. Create attention module
            attn_module, vllm_config = _create_attention_module(
                model_path=model_path,
                max_seq_len=seq_len,
                max_batch_size=batch_size,
                is_context=is_context,
                device=device,
            )
        except Exception as e:
            print(f"  Module creation failed: {e}")
            traceback.print_exc()
            return None

        hf_config_ref = vllm_config.model_config.hf_config
        num_heads = hf_config_ref.num_attention_heads

        # 1b. Process weights
        try:
            _process_module_weights(attn_module, vllm_config)
        except Exception as e:
            print(f"  Weight processing failed (non-fatal): {e}")

        # 2. KV cache + metadata
        try:
            with set_current_vllm_config(vllm_config):
                kv_cache_tuple, attn_metadata = _create_kv_cache_and_metadata(
                    vllm_config=vllm_config,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    is_context=is_context,
                    device=device,
                )
        except Exception as e:
            print(f"  KV cache creation failed: {e}")
            traceback.print_exc()
            _cleanup()
            return None

        # 3. Bind KV cache to forward context.
        # AscendSFAImpl.forward() receives kv_cache as a 3-tuple
        # (kv_nope, kv_pe, k_li); the indexer cache is bundled in the tuple,
        # not registered separately as on CUDA.
        attn_layer_name = "model.layers.0.self_attn.attn"
        forward_ctx = vllm_config.compilation_config.static_forward_context
        if attn_layer_name in forward_ctx:
            forward_ctx[attn_layer_name].kv_cache = [kv_cache_tuple]

        # 4. Input tensors
        hidden_size = hf_config_ref.hidden_size
        if is_context:
            num_tokens = seq_len * batch_size
            positions = (
                torch.arange(seq_len, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
                .contiguous()
            )
        else:
            num_tokens = batch_size
            positions = torch.full(
                (batch_size,), seq_len - 1, device=device, dtype=torch.long
            )

        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )

        # 5. Set forward context
        attn_metadata_dict = {attn_layer_name: attn_metadata}
        stack.enter_context(set_current_vllm_config(vllm_config))
        stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))

        # 6. Dry run
        try:
            with torch.inference_mode():
                attn_module.forward(positions, hidden_states, None)
        except Exception as e:
            print(f"  Dry run failed: {e}")
            traceback.print_exc()
            _cleanup()
            return None

        # 7. Benchmark
        def kernel_func():
            attn_module.forward(positions, hidden_states, None)

        result: BenchResult = benchmark_npu(
            kernel_func,
            warmup_iters=warmup_iters,
            num_runs=bench_iters,
        )
        latency_ms = result.avg_us / 1000.0

    # 8. Write output row
    architecture = getattr(
        hf_config_ref,
        "architectures",
        [getattr(hf_config_ref, "model_type", "unknown")],
    )[0]

    if is_context:
        isl = seq_len
        step = 0
        fname = "dsa_context_module_perf.txt"
        columns = [
            "framework", "version", "device", "op_name", "kernel_source",
            "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
            "kv_cache_dtype", "architecture", "latency",
        ]
        row = {
            "framework": "vllm-ascend",
            "version": "0.18.0",
            "device": "Ascend 910B",
            "op_name": "dsa_context_module",
            "kernel_source": "vllm_ascend_mla",
            "batch_size": batch_size,
            "isl": isl,
            "num_heads": num_heads,
            "gemm_type": "float16",
            "mla_dtype": "float16",
            "kv_cache_dtype": "float16",
            "architecture": architecture,
            "latency": f"{latency_ms:.6f}",
        }
    else:
        fname = "dsa_generation_module_perf.txt"
        columns = [
            "framework", "version", "device", "op_name", "kernel_source",
            "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
            "kv_cache_dtype", "architecture", "step", "latency",
        ]
        row = {
            "framework": "vllm-ascend",
            "version": "0.18.0",
            "device": "Ascend 910B",
            "op_name": "dsa_generation_module",
            "kernel_source": "vllm_ascend_mla",
            "batch_size": batch_size,
            "isl": 1,
            "num_heads": num_heads,
            "gemm_type": "float16",
            "mla_dtype": "float16",
            "kv_cache_dtype": "float16",
            "architecture": architecture,
            "step": seq_len - 1,
            "latency": f"{latency_ms:.6f}",
        }

    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / fname
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}: "
        f"{latency_ms:.4f} ms"
    )
    _cleanup()
    return latency_ms


def _cleanup():
    torch.npu.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    from gemm_factory import _init_vllm_context

    parser = argparse.ArgumentParser(
        description="DSA module-level collector for Ascend NPU (Method C)"
    )
    parser.add_argument(
        "--mode", choices=["context", "generation"], required=True,
        help="Prefill (context) or decode (generation) phase",
    )
    parser.add_argument(
        "--model", type=str, default="zai-org/GLM-5",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size (--quick)")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq len (--quick)")
    parser.add_argument("--quick", action="store_true", help="Single-point quick test")
    parser.add_argument("--output-dir", type=str, default="./dsa_module_data")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    args = parser.parse_args()

    print("Initializing vLLM + Ascend context...")
    _init_vllm_context()

    output_dir = Path(args.output_dir)

    if args.quick:
        b = args.batch_size or 4
        s = args.seq_len or 2048
        run_dsa_module(
            seq_len=s,
            batch_size=b,
            model_path=args.model,
            mode=args.mode,
            output_dir=output_dir,
            device=args.device,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        return

    if args.mode == "context":
        test_cases = get_context_test_cases()
    else:
        test_cases = get_generation_test_cases()

    # Filter to requested model
    test_cases = [(s, b, m) for s, b, m in test_cases if m == args.model]
    print(f"Running {len(test_cases)} {args.mode} DSA module test cases...")

    for i, (s, b, model_path) in enumerate(test_cases):
        print(f"[{i + 1}/{len(test_cases)}]", end="")
        try:
            run_dsa_module(
                seq_len=s,
                batch_size=b,
                model_path=model_path,
                mode=args.mode,
                output_dir=output_dir,
                device=args.device,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
        except Exception as e:
            print(f"  FAILED b={b}, s={s}: {e}")
            traceback.print_exc()
            _cleanup()


if __name__ == "__main__":
    main()
