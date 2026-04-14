"""MLA microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from AIConfigurator's collect_attn.py design with MLA specifics.
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
    import torch_npu
except ImportError:
    pass

from bench_engine import BenchResult, benchmark_npu
from gemm_factory import _init_vllm_context

from mla_factory import (
    BLOCK_SIZE,
    OP_CONTEXT,
    OP_GENERATION,
    SUPPORTED_OP_TYPES,
    MlaSpec,
    create_mla_func,
)

logger = logging.getLogger(__name__)

# --- Default parameter space ---
DEFAULT_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_SEQ_CONTEXT = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_SEQ_GENERATION = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

KERNEL_TYPE_MAP = {
    OP_CONTEXT: "FusedInferAttentionScore_MLA",
    OP_GENERATION: "FusedInferAttentionScore_Decode_MLA",
}

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
    "KV LoRA Rank",
    "QK Nope Dim",
    "QK Rope Dim",
]

CHECKPOINT_FILE = "mla_checkpoint.json"


def _format_context_shapes(spec: MlaSpec) -> tuple[str, str]:
    num_tokens = spec.batch * spec.seq_len
    # Format: Q, kv_c, k_pe -> Output
    q_shape = f"{num_tokens},{spec.num_heads},{spec.head_size}"
    kv_c_shape = f"{num_tokens},{spec.kv_lora_rank}"
    k_pe_shape = f"{num_tokens},1,{spec.qk_rope_head_dim}"
    input_shapes = f"{q_shape};{kv_c_shape};{k_pe_shape}"
    output_shapes = f"{num_tokens},{spec.num_heads},{spec.v_head_dim}"
    return input_shapes, output_shapes


def _format_generation_shapes(spec: MlaSpec) -> tuple[str, str]:
    import math
    blocks_per_seq = math.ceil(spec.seq_len / BLOCK_SIZE)
    num_blocks = spec.batch * blocks_per_seq
    
    q_shape = f"{spec.batch},{spec.num_heads},{spec.head_size}"
    kv_c_shape = f"{spec.batch},{spec.kv_lora_rank}"
    k_pe_shape = f"{spec.batch},1,{spec.qk_rope_head_dim}"
    # Representing the joined MLA cache size
    cache_shape = f"{num_blocks},{BLOCK_SIZE},{spec.kv_cache_head_size}"
    
    input_shapes = f"{q_shape};{kv_c_shape};{k_pe_shape};{cache_shape}"
    output_shapes = f"{spec.batch},{spec.num_heads},{spec.v_head_dim}"
    return input_shapes, output_shapes


def _make_csv_row(spec: MlaSpec, result: BenchResult) -> dict[str, str]:
    if spec.op_type == OP_CONTEXT:
        input_shapes, output_shapes = _format_context_shapes(spec)
    else:
        input_shapes, output_shapes = _format_generation_shapes(spec)

    return {
        "OP State": "dynamic",
        "Accelerator Core": "AI_CORE",
        "Input Shapes": input_shapes,
        "Input Data Types": "DT_BF16",
        "Input Formats": "ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": f"{result.avg_us:.2f}",
        "Op Type": spec.op_type,
        "Batch": str(spec.batch),
        "Seq Len": str(spec.seq_len),
        "Num Heads": str(spec.num_heads),
        "KV LoRA Rank": str(spec.kv_lora_rank),
        "QK Nope Dim": str(spec.qk_nope_head_dim),
        "QK Rope Dim": str(spec.qk_rope_head_dim),
    }


def _spec_key(spec: MlaSpec) -> str:
    return (
        f"{spec.op_type}_{spec.batch}_{spec.seq_len}"
        f"_{spec.num_heads}_{spec.kv_lora_rank}"
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
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> list[MlaSpec]:
    specs: list[MlaSpec] = []
    for op_type in op_types:
        for batch in batch_list:
            for seq_len in seq_len_list:
                for num_heads in num_heads_list:
                    specs.append(MlaSpec(
                        op_type=op_type,
                        batch=batch,
                        seq_len=seq_len,
                        num_heads=num_heads,
                        kv_lora_rank=kv_lora_rank,
                        qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=qk_rope_head_dim,
                        v_head_dim=v_head_dim,
                    ))
    return specs


def run_benchmark(
    specs: list[MlaSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
) -> None:
    device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

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
            "%s %s batch=%d seq=%d heads=%d kv_rank=%d",
            progress, spec.op_type, spec.batch, spec.seq_len,
            spec.num_heads, spec.kv_lora_rank,
        )

        try:
            attn_func = create_mla_func(spec, device)
            result = benchmark_npu(
                attn_func,
                warmup_iters=warmup_iters,
                num_runs=bench_iters,
            )
            row = _make_csv_row(spec, result)
            _, writer = csv_files[spec.op_type]
            writer.writerow(row)

            logger.info(
                "%s -> %.2f us (avg of %d runs)",
                progress, result.avg_us, result.num_runs,
            )
        except Exception:
            logger.exception(
                "%s FAILED %s batch=%d seq=%d heads=%d",
                progress, spec.op_type, spec.batch, spec.seq_len,
                spec.num_heads,
            )
            errors += 1
            continue
        finally:
            try:
                del attn_func
            except NameError:
                pass
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
        description="MLA microbenchmark collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mla_data",
        help="Output directory for CSV files and checkpoint",
    )
    parser.add_argument(
        "--op-types", nargs="+", default=["context", "generation"],
        choices=["context", "generation"],
        help="Op types to benchmark",
    )
    parser.add_argument(
        "--batch-list", nargs="+", type=int, default=None,
    )
    parser.add_argument(
        "--seq-len-list", nargs="+", type=int, default=None,
    )
    parser.add_argument(
        "--num-heads-list", nargs="+", type=int, default=[128], # default DSV3 heads
    )
    parser.add_argument(
        "--kv-lora-rank", type=int, default=512,
    )
    parser.add_argument(
        "--qk-nope-head-dim", type=int, default=128,
    )
    parser.add_argument(
        "--qk-rope-head-dim", type=int, default=64,
    )
    parser.add_argument(
        "--v-head-dim", type=int, default=128,
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=20,
    )
    parser.add_argument(
        "--bench-iters", type=int, default=100,
    )
    parser.add_argument(
        "--resume", action="store_true",
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

    op_type_map = {"context": OP_CONTEXT, "generation": OP_GENERATION}
    op_types = [op_type_map[o] for o in args.op_types]

    batch_list = args.batch_list or DEFAULT_BATCH_LIST
    num_heads_list = args.num_heads_list

    all_specs: list[MlaSpec] = []
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
            kv_lora_rank=args.kv_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
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
