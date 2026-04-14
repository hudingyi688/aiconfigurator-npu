"""GEMM microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from AIConfigurator's collect_gemm.py design:
- Exhaustive (M, N, K) parameter space sweep
- Real kernel path via vllm-ascend operator instantiation
- NPU Event timing with NPU Graph capture + replay (no power sampling)
- CSV output aligned with TensorCast profiling database format

Usage:
    # BF16 only, small shape set
    python collect_gemm.py --quant-types bf16 --output-dir ./gemm_data

    # BF16 + W8A8, full sweep
    python collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./gemm_data

    # Custom M/N/K ranges
    python collect_gemm.py --m-list 1 16 128 1024 --n-list 4096 8192 --k-list 4096 8192

    # Resume from checkpoint
    python collect_gemm.py --output-dir ./gemm_data --resume
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import TextIO

import torch

try:
    import torch_npu  # noqa: F401 — register NPU dispatch
except ImportError:
    pass

from bench_engine import BenchResult, benchmark_npu
from gemm_factory import (
    QUANT_BF16,
    QUANT_W8A8_DYNAMIC,
    SUPPORTED_QUANT_TYPES,
    GemmSpec,
    _init_vllm_context,
    create_gemm_func,
)

logger = logging.getLogger(__name__)

# --- Default parameter space (subset of AIConfigurator's ~97K combos) ---
# M: small values dense (decode), large values sparse (prefill)
DEFAULT_M_LIST = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
]
# N, K: common LLM hidden dimensions
DEFAULT_NK_LIST = [
    256, 512, 1024, 2048, 4096, 7168, 8192, 12288, 16384,
]

# --- CSV output file names (must match op_mapping.yaml kernel_type) ---
KERNEL_TYPE_MAP = {
    QUANT_BF16: "MatMulV2",
    QUANT_W8A8_DYNAMIC: "QuantBatchMatmulV3",
}

# --- CSV column names (aligned with TensorCast profiling database) ---
CSV_COLUMNS = [
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Average Duration(us)",
    "Quant Type",
]

CHECKPOINT_FILE = "checkpoint.json"


def _dtype_str(quant_type: str) -> tuple[str, str]:
    """Return (input_dtype_str, weight_dtype_str) for CSV.

    Must match the actual NPU profiling kernel_details format:
      BF16:  input=DT_BF16, weight=DT_BF16
      W8A8:  input=INT8 (after npu_dynamic_quant), weight=INT8
    """
    if quant_type == QUANT_BF16:
        return "DT_BF16", "DT_BF16"
    if quant_type == QUANT_W8A8_DYNAMIC:
        return "INT8", "INT8"
    raise ValueError(f"Unknown quant_type: {quant_type}")


def _format_shapes(m: int, n: int, k: int) -> tuple[str, str]:
    """Format Input Shapes and Output Shapes for CSV.

    Input: two matrices [M,K] and [N,K] (weight transposed internally).
    Output: [M,N].
    """
    input_shapes = f"{m},{k};{n},{k}"
    output_shapes = f"{m},{n}"
    return input_shapes, output_shapes


def _make_csv_row(spec: GemmSpec, result: BenchResult, op_count: int) -> dict[str, str]:
    """Build a CSV row dict from spec and benchmark result.

    latency = result.avg_us / op_count (single GEMM latency from multi-op run).
    """
    input_dt, weight_dt = _dtype_str(spec.quant_type)
    input_shapes, output_shapes = _format_shapes(spec.m, spec.n, spec.k)
    single_gemm_us = result.avg_us / op_count

    if spec.quant_type == QUANT_W8A8_DYNAMIC:
        accelerator_core = "MIX_AIC"
    else:
        accelerator_core = "AI_CORE"

    return {
        "OP State": "dynamic",
        "Accelerator Core": accelerator_core,
        "Input Shapes": input_shapes,
        "Input Data Types": f"{input_dt};{weight_dt}",
        "Input Formats": "ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": f"{single_gemm_us:.2f}",
        "Quant Type": spec.quant_type,
    }


def _spec_key(spec: GemmSpec) -> str:
    """Unique string key for checkpoint tracking."""
    return f"{spec.m}_{spec.n}_{spec.k}_{spec.quant_type}"


def _load_checkpoint(output_dir: Path) -> set[str]:
    """Load completed spec keys from checkpoint file."""
    ckpt_path = output_dir / CHECKPOINT_FILE
    if not ckpt_path.exists():
        return set()
    with open(ckpt_path) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_checkpoint(output_dir: Path, completed: set[str]) -> None:
    """Atomically save checkpoint."""
    ckpt_path = output_dir / CHECKPOINT_FILE
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp_path.replace(ckpt_path)


def _build_spec_list(
    m_list: list[int],
    n_list: list[int],
    k_list: list[int],
    quant_types: list[str],
) -> list[GemmSpec]:
    """Build cartesian product of (M, N, K, quant_type)."""
    specs = []
    for qt in quant_types:
        for m in m_list:
            for n in n_list:
                for k in k_list:
                    specs.append(GemmSpec(m=m, n=n, k=k, quant_type=qt))
    return specs


def run_benchmark(
    specs: list[GemmSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
) -> None:
    """Run GEMM benchmark for all specs, writing results to CSV."""
    device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint support
    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    # Separate CSV per quant type (aligned with profiling database convention)
    quant_types_in_specs = sorted({s.quant_type for s in specs})
    csv_files: dict[str, tuple[TextIO, csv.DictWriter]] = {}

    for qt in quant_types_in_specs:
        kernel_type = KERNEL_TYPE_MAP[qt]
        csv_path = output_dir / f"{kernel_type}.csv"
        file_exists = csv_path.exists() and resume
        fh = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        csv_files[qt] = (fh, writer)

    total = len(specs)
    skipped = 0
    errors = 0
    t_start = time.monotonic()

    for i, spec in enumerate(specs):
        key = _spec_key(spec)
        if key in completed:
            skipped += 1
            continue

        progress = f"[{i + 1}/{total}]"
        logger.info(
            "%s M=%d N=%d K=%d quant=%s",
            progress, spec.m, spec.n, spec.k, spec.quant_type,
        )

        try:
            gemm_func, op_count = create_gemm_func(spec, device)
            result = benchmark_npu(
                gemm_func,
                warmup_iters=warmup_iters,
                num_runs=bench_iters,
            )
            row = _make_csv_row(spec, result, op_count)
            _, writer = csv_files[spec.quant_type]
            writer.writerow(row)

            single_us = result.avg_us / op_count
            graph_tag = "graph" if result.used_graph else "eager"
            logger.info(
                "%s -> %.2f us (avg of %d runs, %d ops/run, %s)",
                progress, single_us, result.num_runs, op_count, graph_tag,
            )
        except Exception:
            logger.exception("%s FAILED M=%d N=%d K=%d quant=%s", progress, spec.m, spec.n, spec.k, spec.quant_type)
            errors += 1
            continue
        finally:
            # Release NPU memory between shapes to avoid OOM on large sweeps
            try:
                del gemm_func
            except NameError:
                pass
            try:
                torch.npu.synchronize()
            except RuntimeError:
                pass
            torch.npu.empty_cache()

        completed.add(key)

        # Flush CSV and checkpoint periodically
        if len(completed) % 10 == 0:
            for fh, _ in csv_files.values():
                fh.flush()
            _save_checkpoint(output_dir, completed)

    # Final flush
    for fh, _ in csv_files.values():
        fh.flush()
        fh.close()
    _save_checkpoint(output_dir, completed)

    elapsed = time.monotonic() - t_start
    logger.info(
        "Done: %d benchmarked, %d skipped, %d errors in %.1fs",
        len(completed) - skipped, skipped, errors, elapsed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEMM microbenchmark collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./gemm_data",
        help="Output directory for CSV files and checkpoint",
    )
    parser.add_argument(
        "--quant-types", nargs="+", default=["bf16"],
        choices=list(SUPPORTED_QUANT_TYPES),
        help="Quantization types to benchmark",
    )
    parser.add_argument(
        "--m-list", nargs="+", type=int, default=None,
        help="M dimensions (default: built-in list)",
    )
    parser.add_argument(
        "--n-list", nargs="+", type=int, default=None,
        help="N dimensions (default: built-in list)",
    )
    parser.add_argument(
        "--k-list", nargs="+", type=int, default=None,
        help="K dimensions (default: built-in list)",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=20,
        help="Warmup iterations per shape",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
        help="Number of timed runs (single Event pair wrapping all runs)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip already-completed shapes)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Initializing vLLM context...")
    _init_vllm_context()

    m_list = args.m_list or DEFAULT_M_LIST
    n_list = args.n_list or DEFAULT_NK_LIST
    k_list = args.k_list or DEFAULT_NK_LIST

    specs = _build_spec_list(m_list, n_list, k_list, args.quant_types)
    logger.info(
        "Parameter space: %d M x %d N x %d K x %d quant = %d total specs",
        len(m_list), len(n_list), len(k_list), len(args.quant_types), len(specs),
    )

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.num_runs,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
