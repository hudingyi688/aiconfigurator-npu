"""Attention microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from AIConfigurator's collect_attn.py design:
- Two op types: context (prefill via FIA) and generation (decode via PA)
- Exhaustive (batch, seq_len, num_heads, num_kv_heads) parameter space sweep
- Real kernel path via vllm-ascend attention operators
- NPU Event timing with NPU Graph capture + replay (no power sampling)
- CSV output aligned with TensorCast profiling database format

Usage:
    # Context + generation, small shape set
    python collect_attn.py --op-types context generation \
        --batch-list 1 4 --seq-len-list 128 1024 \
        --num-heads-list 32 --num-kv-heads-list 0 8

    # Full sweep
    python collect_attn.py --op-types context generation --output-dir ./attn_data

    # Resume from checkpoint
    python collect_attn.py --output-dir ./attn_data --resume
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
    import torch_npu  # noqa: F401
except ImportError:
    pass

from attn_factory import (
    BLOCK_SIZE,
    OP_CONTEXT,
    OP_GENERATION,
    SUPPORTED_OP_TYPES,
    AttnSpec,
    _resolve_kv_heads,
    create_attn_func,
)
from bench_engine import BenchResult, benchmark_npu
from gemm_factory import _init_vllm_context

logger = logging.getLogger(__name__)

# --- Default parameter space ---
DEFAULT_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_SEQ_CONTEXT = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_SEQ_GENERATION = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_NUM_HEADS = [8, 16, 32, 40, 64]
DEFAULT_NUM_KV_HEADS = [0, 1, 4, 8]  # 0 = MHA

# --- CSV output file names (must match op_mapping.yaml kernel_type) ---
# Both context and generation use FIA on Ascend (_npu_paged_attention is
# only used under specific graph capture conditions, not the general path)
KERNEL_TYPE_MAP = {
    OP_CONTEXT: "FusedInferAttentionScore",
    OP_GENERATION: "FusedInferAttentionScore_Decode",
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
    "Op Type",
    "Batch",
    "Seq Len",
    "Num Heads",
    "Num KV Heads",
    "Head Size",
]

CHECKPOINT_FILE = "attn_checkpoint.json"


def _format_context_shapes(spec: AttnSpec) -> tuple[str, str]:
    """Format Input/Output Shapes for context (prefill) CSV row.

    TND layout: Q=[num_tokens, num_heads, head_size], K/V=[num_tokens, num_kv_heads, head_size]
    """
    num_kv_heads = _resolve_kv_heads(spec)
    num_tokens = spec.batch * spec.seq_len
    q_shape = f"{num_tokens},{spec.num_heads},{spec.head_size}"
    kv_shape = f"{num_tokens},{num_kv_heads},{spec.head_size}"
    input_shapes = f"{q_shape};{kv_shape};{kv_shape}"
    output_shapes = f"{num_tokens},{spec.num_heads},{spec.head_size}"
    return input_shapes, output_shapes


def _format_generation_shapes(spec: AttnSpec) -> tuple[str, str]:
    """Format Input/Output Shapes for generation (decode) CSV row.

    FIA decode: Q=[batch, num_heads, head_size],
    K/V=[num_blocks, block_size, num_kv_heads * head_size] (flattened for FIA)
    """
    import math

    num_kv_heads = _resolve_kv_heads(spec)
    blocks_per_seq = math.ceil(spec.seq_len / BLOCK_SIZE)
    num_blocks = spec.batch * blocks_per_seq
    hidden = num_kv_heads * spec.head_size
    q_shape = f"{spec.batch},{spec.num_heads},{spec.head_size}"
    kv_shape = f"{num_blocks},{BLOCK_SIZE},{hidden}"
    input_shapes = f"{q_shape};{kv_shape};{kv_shape}"
    output_shapes = f"{spec.batch},{spec.num_heads},{spec.head_size}"
    return input_shapes, output_shapes


def _make_csv_row(spec: AttnSpec, result: BenchResult) -> dict[str, str]:
    """Build a CSV row dict from spec and benchmark result."""
    if spec.op_type == OP_CONTEXT:
        input_shapes, output_shapes = _format_context_shapes(spec)
    else:
        input_shapes, output_shapes = _format_generation_shapes(spec)

    return {
        "OP State": "dynamic",
        "Accelerator Core": "AI_CORE",
        "Input Shapes": input_shapes,
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
        "Input Formats": "ND;ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": f"{result.avg_us:.2f}",
        "Op Type": spec.op_type,
        "Batch": str(spec.batch),
        "Seq Len": str(spec.seq_len),
        "Num Heads": str(spec.num_heads),
        "Num KV Heads": str(spec.num_kv_heads),
        "Head Size": str(spec.head_size),
    }


def _spec_key(spec: AttnSpec) -> str:
    """Unique string key for checkpoint tracking."""
    return (
        f"{spec.op_type}_{spec.batch}_{spec.seq_len}"
        f"_{spec.num_heads}_{spec.num_kv_heads}"
    )


def _load_checkpoint(output_dir: Path) -> set[str]:
    ckpt_path = output_dir / CHECKPOINT_FILE
    if not ckpt_path.exists():
        return set()
    with open(ckpt_path) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_checkpoint(output_dir: Path, completed: set[str]) -> None:
    ckpt_path = output_dir / CHECKPOINT_FILE
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp_path.replace(ckpt_path)


def _build_spec_list(
    op_types: list[str],
    batch_list: list[int],
    seq_len_list: list[int],
    num_heads_list: list[int],
    num_kv_heads_list: list[int],
    head_size: int,
) -> list[AttnSpec]:
    """Build cartesian product with GQA constraint validation."""
    specs: list[AttnSpec] = []
    for op_type in op_types:
        for batch in batch_list:
            for seq_len in seq_len_list:
                for num_heads in num_heads_list:
                    for num_kv_heads in num_kv_heads_list:
                        # Resolve MHA: 0 -> num_heads
                        effective_kv = num_heads if num_kv_heads == 0 else num_kv_heads
                        # GQA constraint: num_heads must be divisible by num_kv_heads
                        if effective_kv > num_heads:
                            continue
                        if num_heads % effective_kv != 0:
                            continue
                        specs.append(AttnSpec(
                            op_type=op_type,
                            batch=batch,
                            seq_len=seq_len,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                        ))
    return specs


def run_benchmark(
    specs: list[AttnSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
) -> None:
    """Run attention benchmark for all specs, writing results to CSV."""
    device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    # Separate CSV per op type
    op_types_in_specs = sorted({s.op_type for s in specs})
    csv_files: dict[str, tuple[TextIO, csv.DictWriter]] = {}

    for op_type in op_types_in_specs:
        kernel_type = KERNEL_TYPE_MAP[op_type]
        csv_path = output_dir / f"{kernel_type}.csv"
        file_exists = csv_path.exists() and resume
        fh = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        csv_files[op_type] = (fh, writer)

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
            "%s %s batch=%d seq=%d heads=%d kv_heads=%d",
            progress, spec.op_type, spec.batch, spec.seq_len,
            spec.num_heads, spec.num_kv_heads,
        )

        try:
            attn_func = create_attn_func(spec, device)
            result = benchmark_npu(
                attn_func,
                warmup_iters=warmup_iters,
                num_runs=bench_iters,
            )
            row = _make_csv_row(spec, result)
            _, writer = csv_files[spec.op_type]
            writer.writerow(row)

            graph_tag = "graph" if result.used_graph else "eager"
            logger.info(
                "%s -> %.2f us (avg of %d runs, %s)",
                progress, result.avg_us, result.num_runs, graph_tag,
            )
        except Exception:
            logger.exception(
                "%s FAILED %s batch=%d seq=%d heads=%d kv_heads=%d",
                progress, spec.op_type, spec.batch, spec.seq_len,
                spec.num_heads, spec.num_kv_heads,
            )
            errors += 1
            continue
        finally:
            try:
                del attn_func
            except NameError:
                pass
            # Synchronize to clear any error state on the NPU stream,
            # then free memory. Without this, a failed kernel leaves the
            # stream in an error state and subsequent Event.elapsed_time()
            # calls fail with "event recorder null".
            try:
                torch.npu.synchronize()
            except RuntimeError:
                pass
            torch.npu.empty_cache()

        completed.add(key)

        if len(completed) % 10 == 0:
            for fh, _ in csv_files.values():
                fh.flush()
            _save_checkpoint(output_dir, completed)

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
        description="Attention microbenchmark collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./attn_data",
        help="Output directory for CSV files and checkpoint",
    )
    parser.add_argument(
        "--op-types", nargs="+", default=["context", "generation"],
        choices=["context", "generation"],
        help="Op types to benchmark (context=prefill, generation=decode)",
    )
    parser.add_argument(
        "--batch-list", nargs="+", type=int, default=None,
        help="Batch sizes (default: built-in list)",
    )
    parser.add_argument(
        "--seq-len-list", nargs="+", type=int, default=None,
        help="Sequence lengths (default: per-op-type built-in list)",
    )
    parser.add_argument(
        "--num-heads-list", nargs="+", type=int, default=None,
        help="Number of attention heads (default: built-in list)",
    )
    parser.add_argument(
        "--num-kv-heads-list", nargs="+", type=int, default=None,
        help="Number of KV heads, 0=MHA (default: built-in list)",
    )
    parser.add_argument(
        "--head-size", type=int, default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=20,
        help="Warmup iterations per shape",
    )
    parser.add_argument(
        "--bench-iters", type=int, default=100,
        help="Benchmark iterations per shape",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
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

    # Map CLI op names to internal constants
    op_type_map = {"context": OP_CONTEXT, "generation": OP_GENERATION}
    op_types = [op_type_map[o] for o in args.op_types]

    batch_list = args.batch_list or DEFAULT_BATCH_LIST
    num_heads_list = args.num_heads_list or DEFAULT_NUM_HEADS
    num_kv_heads_list = args.num_kv_heads_list or DEFAULT_NUM_KV_HEADS

    # Build specs per op type with appropriate seq_len defaults
    all_specs: list[AttnSpec] = []
    for op_type in op_types:
        if args.seq_len_list:
            seq_list = args.seq_len_list
        elif op_type == OP_CONTEXT:
            seq_list = DEFAULT_SEQ_CONTEXT
        else:
            seq_list = DEFAULT_SEQ_GENERATION

        specs = _build_spec_list(
            op_types=[op_type],
            batch_list=batch_list,
            seq_len_list=seq_list,
            num_heads_list=num_heads_list,
            num_kv_heads_list=num_kv_heads_list,
            head_size=args.head_size,
        )
        all_specs.extend(specs)

    logger.info("Total specs: %d", len(all_specs))

    run_benchmark(
        specs=all_specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
