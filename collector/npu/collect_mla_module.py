"""DSA Module Collector for Ascend NPU — Module-level (Method C).

Profiles the complete DeepseekV2MLAAttention module forward pass
(projections + MLA attention + output), not just the bare attention kernel.

Output: dsa_context_module_perf.txt / dsa_generation_module_perf.txt
        (aiconfigurator DSA module format, ready for config search)

Usage:
    python collector/npu/collect_mla_module.py --mode context
    python collector/npu/collect_mla_module.py --mode generation
    python collector/npu/collect_mla_module.py --mode context --quick --batch-size 4 --seq-len 2048
"""

import argparse
import csv
import gc
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def _ensure_c_ascend_loaded() -> None:
    """Force-load vllm_ascend_C.so so torch.ops._C_ascend.* are available.

    vllm-ascend only dlopens this extension lazily via
    `from vllm_ascend.vllm_ascend_C import init_module` inside
    device_allocator/camem.py, wrapped in try/except. If that import
    fails, the extension — and every op registered via
    TORCH_LIBRARY(_C_ascend, ...) — silently never loads. The DSA
    collector needs mla_preprocess / npu_sparse_flash_attention /
    npu_lightning_indexer[_quant], so we load it explicitly.
    """
    import glob
    import os

    try:
        import vllm_ascend  # type: ignore
    except ImportError:
        return

    root = os.path.dirname(vllm_ascend.__file__)
    candidates = sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so")))
    for so in candidates:
        try:
            torch.ops.load_library(so)
        except Exception as e:
            print(f"[WARN] failed to load {so}: {e}")


_ensure_c_ascend_loaded()

from bench_engine import BenchResult, benchmark_npu
from gemm_factory import _init_vllm_context
from mla_module_factory import (
    OP_CONTEXT,
    OP_GENERATION,
    DsaModuleSpec,
    create_dsa_module_func,
)

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


def get_context_test_cases(model_path: str) -> list[tuple[int, int]]:
    """Return (seq_len, batch_size) tuples for context phase."""
    return [
        (s, b)
        for b in _CONTEXT_BATCH_LIST
        for s in _CONTEXT_SEQ_LIST
        if b * s <= 131072
    ]


def get_generation_test_cases(model_path: str) -> list[tuple[int, int]]:
    """Return (kv_cache_len, batch_size) tuples for generation phase."""
    return [
        (s, b)
        for b in _GENERATION_BATCH_LIST
        for s in _GENERATION_SEQ_LIST
        if b * s <= 1024 * 4096 * 8
    ]


# ═══════════════════════════════════════════════════════════════════════
# Output writing
# ═══════════════════════════════════════════════════════════════════════

_CONTEXT_COLUMNS = [
    "framework", "version", "device", "op_name", "kernel_source",
    "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
    "kv_cache_dtype", "architecture", "latency",
]
_GENERATION_COLUMNS = [
    "framework", "version", "device", "op_name", "kernel_source",
    "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
    "kv_cache_dtype", "architecture", "step", "latency",
]


def _write_row(
    output_dir: Path,
    is_context: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    architecture: str,
    latency_ms: float,
) -> None:
    if is_context:
        fname = "dsa_context_module_perf.txt"
        columns = _CONTEXT_COLUMNS
        row = {
            "framework": "vllm-ascend",
            "version": "0.18.0",
            "device": "Ascend 910B",
            "op_name": "dsa_context_module",
            "kernel_source": "vllm_ascend_mla",
            "batch_size": batch_size,
            "isl": seq_len,
            "num_heads": num_heads,
            "gemm_type": "float16",
            "mla_dtype": "float16",
            "kv_cache_dtype": "float16",
            "architecture": architecture,
            "latency": f"{latency_ms:.6f}",
        }
    else:
        fname = "dsa_generation_module_perf.txt"
        columns = _GENERATION_COLUMNS
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

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / fname
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    op_type = OP_CONTEXT if is_context else OP_GENERATION
    phase = "context" if is_context else "generation"

    print(f"\n[DSA module] {phase} b={batch_size}, s={seq_len}, model={model_path}")

    spec = DsaModuleSpec(
        op_type=op_type,
        batch=batch_size,
        seq_len=seq_len,
        model_path=model_path,
    )

    try:
        forward_fn, meta = create_dsa_module_func(spec, device=device)
    except Exception as e:
        print(f"  Setup failed: {e}")
        traceback.print_exc()
        _cleanup(device)
        return None

    stack = meta["_stack"]
    num_heads = meta["num_heads"]
    hf_config = meta["hf_config"]
    architecture = getattr(
        hf_config, "architectures",
        [getattr(hf_config, "model_type", "unknown")],
    )[0]

    try:
        result: BenchResult = benchmark_npu(
            forward_fn,
            warmup_iters=warmup_iters,
            num_runs=bench_iters,
        )
        latency_ms = result.avg_us / 1000.0
    except Exception as e:
        print(f"  Benchmark failed: {e}")
        traceback.print_exc()
        stack.close()
        _cleanup(device)
        return None
    finally:
        stack.close()

    _write_row(
        output_dir=output_dir,
        is_context=is_context,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        architecture=architecture,
        latency_ms=latency_ms,
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}: "
        f"{latency_ms:.4f} ms"
    )
    _cleanup(device)
    return latency_ms


def _cleanup(device: str = "npu:0") -> None:
    torch.npu.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="DSA module-level collector for Ascend NPU (Method C)"
    )
    parser.add_argument(
        "--mode", choices=["context", "generation"], required=True,
        help="Prefill (context) or decode (generation) phase",
    )
    parser.add_argument(
        "--model", type=str, default="zai-org/GLM-5",
        help="Model to benchmark (HuggingFace name or local path)",
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
        run_dsa_module(
            seq_len=args.seq_len or 2048,
            batch_size=args.batch_size or 4,
            model_path=args.model,
            mode=args.mode,
            output_dir=output_dir,
            device=args.device,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        return

    if args.mode == "context":
        test_cases = get_context_test_cases(args.model)
    else:
        test_cases = get_generation_test_cases(args.model)

    print(f"Running {len(test_cases)} {args.mode} DSA module test cases...")

    for i, (s, b) in enumerate(test_cases):
        print(f"[{i + 1}/{len(test_cases)}]", end="")
        try:
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
        except Exception as e:
            print(f"  FAILED b={b}, s={s}: {e}")
            traceback.print_exc()
            _cleanup(args.device)


if __name__ == "__main__":
    main()
