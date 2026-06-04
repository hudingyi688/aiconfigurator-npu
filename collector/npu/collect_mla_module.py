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
import os
import traceback
from pathlib import Path


def _ensure_oot_custom_opp_path() -> None:
    """Prepend vllm-ascend's OOT custom-op vendor dir to ASCEND_CUSTOM_OPP_PATH.

    vllm-ascend ships its own SparseFlashAttention bin (5 attrs) under
    `_cann_ops_custom/vendors/vllm-ascend/`. CANN ships another one
    under `opp/built-in/.../sparse_flash_attention/` (9 attrs). The
    op def in vllm_ascend_C.so registers 5 attrs, so it MUST dispatch
    to the OOT bin, otherwise the kernel reads attr index 5/6/7/8
    out of range 5 and segfaults.

    `NPUPlatform.import_kernels()` would set this variable too, but
    it runs AFTER `import torch_npu` and ACL has already cached the
    dispatch search path by then. Real LLM inference works because
    the path is exported in the shell BEFORE python launches; the
    collector skips that startup hook so we set it here, before any
    torch import.

    set_env.bash that ships with vllm-ascend hardcodes
    `/usr/local/package/vllm-ascend/...` which is the upstream's
    intended install prefix — wrong for any pip-installed setup.
    Compute the real path from the importable `vllm_ascend` instead.
    """
    if os.environ.get("AIC_SKIP_OOT_PATH_FIX") in {"1", "true", "TRUE"}:
        return
    try:
        # NOTE: import vllm_ascend before torch_npu so we don't pin the
        # ACL custom-opp search path before this var is set.
        import vllm_ascend  # type: ignore
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


_ensure_oot_custom_opp_path()


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


def _ensure_npu_compile_opts() -> None:
    """Apply NPU init sequence that model_runner_v1 / worker would set.

    AIConfigurator builds vllm_config manually and skips engine startup,
    so ACL_PRECISION_MODE / ACL_OP_JIT_COMPILE stay unconfigured and the
    current NPU device is never bound. When
    process_weights_after_loading calls npu_format_cast(wd_qkv, 29) or
    any NPU op, ACL then errors with 500001 SetPrecisionMode /
    ACL_OP_JIT_COMPILE.

    Mirrors vllm_ascend/worker/worker.py:_init_device():
      1) torch.npu.set_device(device)
      2) import torch_npu._inductor (warms ACL compile-opt init)
      3) torch.npu.config.allow_internal_format = True
      4) torch.npu.set_compile_mode(jit_compile=False)
      5) trivial NPU op + empty_cache to finish ACL lazy-init
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return

    # IMPORTANT: allow_internal_format must be set BEFORE set_device,
    # otherwise ACL context gets initialized with internal_format=False
    # and later toggling is silently ignored (producing the warning
    # "Cannot create tensor with internal format while
    # allow_internel_format=False"). This matches model_runner_v1.py:156
    # which sets it at module import time, long before any set_device().
    try:
        torch.npu.config.allow_internal_format = True
        print("[NPU INIT] allow_internal_format=True OK (pre-set_device)", flush=True)
    except Exception as e:
        print(f"[NPU INIT] allow_internal_format failed: {type(e).__name__}: {e}", flush=True)

    # Verbose so we can see exactly which step trips when things go wrong.
    try:
        torch.npu.set_device(0)
        print("[NPU INIT] set_device(0) OK", flush=True)
    except Exception as e:
        print(f"[NPU INIT] set_device failed: {type(e).__name__}: {e}", flush=True)

    try:
        import torch_npu._inductor  # noqa: F401
        print("[NPU INIT] torch_npu._inductor imported", flush=True)
    except Exception as e:
        print(f"[NPU INIT] torch_npu._inductor import failed: {type(e).__name__}: {e}", flush=True)

    try:
        torch.npu.set_compile_mode(jit_compile=False)
        print("[NPU INIT] set_compile_mode(jit_compile=False) OK", flush=True)
    except Exception as e:
        print(f"[NPU INIT] set_compile_mode failed: {type(e).__name__}: {e}", flush=True)

    try:
        import gc
        t = torch.zeros(1, device="npu:0")
        t = t + 1
        torch.npu.synchronize()
        del t
        gc.collect()
        torch.npu.empty_cache()
        print("[NPU INIT] warmup op + empty_cache OK", flush=True)
    except Exception as e:
        print(f"[NPU INIT] warmup failed: {type(e).__name__}: {e}", flush=True)

    # Probe npu_format_cast(29) here so we catch the real failure at
    # init time instead of deep inside process_weights_after_loading.
    # Check .storage().size() or internal format to confirm NZ took effect.
    try:
        probe = torch.randn(16, 32, dtype=torch.bfloat16, device="npu:0")
        nz = torch_npu.npu_format_cast(probe, 29)
        torch.npu.synchronize()
        nz_format = torch_npu.get_npu_format(nz) if hasattr(torch_npu, "get_npu_format") else "unknown"
        print(f"[NPU INIT] npu_format_cast(29) probe OK shape={tuple(nz.shape)} format={nz_format}", flush=True)
    except Exception as e:
        print(f"[NPU INIT] npu_format_cast(29) probe failed: {type(e).__name__}: {e}", flush=True)


_ensure_c_ascend_loaded()
_ensure_npu_compile_opts()

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
    3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 32768,
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
    gemm_type: str = "float16",
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
            "gemm_type": gemm_type,
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
            "gemm_type": gemm_type,
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


def _load_completed_keys(output_dir: Path, is_context: bool) -> set[tuple[int, int]]:
    """Read the perf .txt and return the set of already-collected keys.

    Used by --resume to skip (b, s) points whose latency row is already
    on disk. Each row covers one (batch_size, seq_len) point — for
    generation rows seq_len is reconstructed as step + 1 since isl
    is hard-coded to 1 in the writer.
    """
    fname = "dsa_context_module_perf.txt" if is_context else "dsa_generation_module_perf.txt"
    path = output_dir / fname
    if not path.exists():
        return set()
    completed: set[tuple[int, int]] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                b = int(row["batch_size"])
                if is_context:
                    s = int(row["isl"])
                else:
                    s = int(row["step"]) + 1
            except (KeyError, ValueError):
                continue
            completed.add((b, s))
    return completed


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
    force_mla: bool = False,
    num_heads_override: int | None = None,
    quantization: str | None = None,
    chunk_query_len: int | None = None,
) -> float | None:
    """Benchmark one (seq_len, batch_size) point for DSA module on NPU."""
    is_context = mode == "context"
    op_type = OP_CONTEXT if is_context else OP_GENERATION
    phase = "context" if is_context else "generation"

    # gemm_type column must match what the model queries at runtime: a w8a8
    # model resolves gemm_quant_mode=w8a8_dynamic, so the row must be tagged
    # accordingly or the perf DB lookup misses (see B-path fix). bf16 baseline
    # stays float16.
    gemm_type = "w8a8_dynamic" if quantization == "ascend" else "float16"

    label = "MLA" if force_mla else "DSA"
    heads_note = f", heads={num_heads_override}" if num_heads_override else ""
    quant_note = f", quant={quantization}" if quantization else ""
    print(f"\n[{label} module] {phase} b={batch_size}, s={seq_len}, model={model_path}{heads_note}{quant_note}")

    spec = DsaModuleSpec(
        op_type=op_type,
        batch=batch_size,
        seq_len=seq_len,
        model_path=model_path,
        force_mla=force_mla,
        num_heads_override=num_heads_override,
        quantization=quantization,
        chunk_query_len=chunk_query_len,
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
        gemm_type=gemm_type,
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
    parser.add_argument(
        "--force-mla",
        action="store_true",
        help="Strip index_topk from hf_config so the platform selector "
             "picks AscendMLABackend instead of AscendSFABackend. "
             "Use this to collect a non-sparse MLA baseline on GLM-5 / "
             "DSA configs while the SparseFlashAttention kernel path is "
             "unstable on synthetic inputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (batch, seq) points whose latency row is already in "
             "the output txt. Useful for resuming a 2-3 day sweep after "
             "a crash without re-running already-collected points.",
    )
    parser.add_argument(
        "--num-heads-override",
        type=int,
        default=None,
        help="Override attention head count. NOTE: GLM-5 DSA PREFILL uses Context "
             "Parallelism (sequence-dim split via all-to-all), NOT tensor-parallel "
             "head split — every rank keeps ALL 64 heads, so use 64 for prefill "
             "regardless of TP (verified: 32-rank tp16 profiler all show SFA nh=64). "
             "DECODE does use head split (tp4 -> 16). The DSA module is a single-rank "
             "op driven by head count; output rows carry this head count and the perf "
             "DB matches on num_heads at query time.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "ascend"],
        help="vllm quantization mode. Omit (None) = bf16 baseline. 'ascend' = "
             "load the W8A8 quant_config from a real w8a8 model dir (pass that "
             "dir via --model, e.g. /mnt/.../GLM-5-w8a8) so the MLA projection "
             "Linears run W8A8 like production; the output gemm_type column "
             "becomes w8a8_dynamic, matching what the w8a8 model queries.",
    )
    parser.add_argument(
        "--chunk-query-len",
        type=int,
        default=None,
        help="Context only. Per-step query window for CHUNKED prefill. Production "
             "value = max_num_batched_tokens / cp_size = 4096 / 16 = 256 (verified: "
             "profiler SFA shape first dim = 256). Models q new tokens attending to "
             "seq_len cumulative KV — matches production sparse flash attention. "
             "Omit for legacy full-seq one-shot prefill (query=seq_len), which "
             "overestimates per-layer DSA ~19x.",
    )
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
            force_mla=args.force_mla,
            num_heads_override=args.num_heads_override,
            quantization=args.quantization,
            chunk_query_len=args.chunk_query_len,
        )
        return

    if args.mode == "context":
        test_cases = get_context_test_cases(args.model)
    else:
        test_cases = get_generation_test_cases(args.model)

    if args.resume:
        completed = _load_completed_keys(output_dir, is_context=args.mode == "context")
        skipped = sum(1 for s, b in test_cases if (b, s) in completed)
        test_cases = [(s, b) for s, b in test_cases if (b, s) not in completed]
        print(f"Resume: {skipped} points already collected, {len(test_cases)} remaining.")

    label = "MLA" if args.force_mla else "DSA"
    print(f"Running {len(test_cases)} {args.mode} {label} module test cases...")

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
                force_mla=args.force_mla,
                num_heads_override=args.num_heads_override,
                quantization=args.quantization,
                chunk_query_len=args.chunk_query_len,
            )
        except Exception as e:
            print(f"  FAILED b={b}, s={s}: {e}")
            traceback.print_exc()
            _cleanup(args.device)


if __name__ == "__main__":
    main()
