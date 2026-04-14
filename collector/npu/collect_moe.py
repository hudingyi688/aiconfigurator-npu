"""MoE microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from AIConfigurator's collect_moe.py design:
- Model-specific parameter space (num_experts, topk, hidden/inter are discrete)
- Real kernel path via vllm-ascend MoE operators
- NPU Event timing with NPU Graph capture + replay (no power sampling)
- CSV output aligned with TensorCast profiling database format

Usage:
    # BF16 only, single model config
    python collect_moe.py --quant-types bf16 --output-dir ./moe_data

    # BF16 + W8A8, full sweep
    python collect_moe.py --quant-types bf16 w8a8_dynamic --output-dir ./moe_data

    # Custom token counts
    python collect_moe.py --token-list 1 16 128 1024

    # Resume from checkpoint
    python collect_moe.py --output-dir ./moe_data --resume
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
from gemm_factory import _init_vllm_context
from moe_factory import (
    QUANT_BF16,
    QUANT_W8A8_DYNAMIC,
    SUPPORTED_QUANT_TYPES,
    MoeSpec,
    create_moe_func,
)

logger = logging.getLogger(__name__)

# --- Default token counts (MoE is model-specific, not exhaustive like GEMM) ---
DEFAULT_TOKEN_LIST = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
]

# --- Model configurations: (hidden, intermediate, num_experts, topk, ep_size, name) ---
# MoE is the only model-specific operator — expert count, topk, hidden/inter
# are discrete architecture params that can't be interpolated.
# ep_size simulates Expert Parallelism: only num_experts/ep_size local experts
# are allocated on a single card, avoiding OOM for large models (e.g. 256 experts).
MODEL_CONFIGS = [
    # DeepSeek-V2-Lite
    (2048, 1408, 64, 6, 1, "deepseek-v2-lite"),
    # DeepSeek-V2
    (5120, 1536, 160, 6, 4, "deepseek-v2"),
    # DeepSeek-V3 / R1
    (7168, 2048, 256, 8, 8, "deepseek-v3"),
    # GLM-5 (744B, 40B active, DSA — same MoE dims as DeepSeek-V3)
    (7168, 2048, 256, 8, 8, "glm-5"),
    # MiniMax-Text-01 (456B, 45.9B active)
    (6144, 9216, 32, 2, 4, "minimax-text-01"),
    # Mixtral-8x7B
    (4096, 14336, 8, 2, 1, "mixtral-8x7b"),
    # Mixtral-8x22B
    (6144, 16384, 8, 2, 1, "mixtral-8x22b"),
    # Qwen2-MoE (57B-A14B)
    (3584, 2560, 64, 8, 1, "qwen2-moe-57b"),
    # Qwen3-MoE (30B-A3B)
    (2048, 1024, 128, 8, 4, "qwen3-moe-30b"),
]

# --- CSV output file names ---
# MoE on NPU uses GroupedMatmul as the core kernel for both BF16 and W8A8.
# W8A8 additionally uses npu_grouped_matmul_swiglu_quant (fused GMM1+SwiGLU).
# We output a single CSV per quant type since the bench measures the full
# MoE forward (routing + GMM1 + SwiGLU + GMM2 + finalize).
KERNEL_TYPE_MAP = {
    QUANT_BF16: "GroupedMatmul_MoE_BF16",
    QUANT_W8A8_DYNAMIC: "GroupedMatmul_MoE_W8A8",
}

# --- CSV column names ---
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
    "Num Tokens",
    "Hidden Size",
    "Intermediate Size",
    "Num Experts",
    "TopK",
    "EP Size",
    "Local Experts",
    "Model",
]

CHECKPOINT_FILE = "moe_checkpoint.json"


def _make_csv_row(
    spec: MoeSpec, result: BenchResult, model_name: str,
) -> dict[str, str]:
    """Build a CSV row dict from spec and benchmark result."""
    if spec.quant_type == QUANT_W8A8_DYNAMIC:
        input_dt = "INT8;INT8"
        accel_core = "MIX_AIC"
    else:
        input_dt = "DT_BF16;DT_BF16"
        accel_core = "AI_CORE"

    # Input shapes: hidden_states [tokens, hidden]; w1 [local_experts, inter*2, hidden]
    input_shapes = (
        f"{spec.num_tokens},{spec.hidden_size};"
        f"{spec.local_num_experts},{spec.intermediate_size * 2},{spec.hidden_size}"
    )
    output_shapes = f"{spec.num_tokens},{spec.hidden_size}"

    return {
        "OP State": "dynamic",
        "Accelerator Core": accel_core,
        "Input Shapes": input_shapes,
        "Input Data Types": input_dt,
        "Input Formats": "ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16",
        "Output Formats": "ND",
        "Average Duration(us)": f"{result.avg_us:.2f}",
        "Quant Type": spec.quant_type,
        "Num Tokens": str(spec.num_tokens),
        "Hidden Size": str(spec.hidden_size),
        "Intermediate Size": str(spec.intermediate_size),
        "Num Experts": str(spec.num_experts),
        "TopK": str(spec.topk),
        "EP Size": str(spec.ep_size),
        "Local Experts": str(spec.local_num_experts),
        "Model": model_name,
    }


def _spec_key(spec: MoeSpec, model_name: str) -> str:
    """Unique string key for checkpoint tracking."""
    return (
        f"{spec.num_tokens}_{spec.hidden_size}_{spec.intermediate_size}"
        f"_{spec.num_experts}_{spec.topk}_{spec.ep_size}"
        f"_{spec.quant_type}_{model_name}"
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
    token_list: list[int],
    model_configs: list[tuple[int, int, int, int, int, str]],
    quant_types: list[str],
) -> list[tuple[MoeSpec, str]]:
    """Build (spec, model_name) pairs from cartesian product."""
    specs: list[tuple[MoeSpec, str]] = []
    for qt in quant_types:
        for hidden, inter, experts, topk, ep_size, name in model_configs:
            for tokens in token_list:
                specs.append((
                    MoeSpec(
                        num_tokens=tokens,
                        hidden_size=hidden,
                        intermediate_size=inter,
                        num_experts=experts,
                        topk=topk,
                        quant_type=qt,
                        ep_size=ep_size,
                    ),
                    name,
                ))
    return specs


def run_benchmark(
    specs: list[tuple[MoeSpec, str]],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
) -> None:
    """Run MoE benchmark for all specs, writing results to CSV."""
    device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    # Separate CSV per quant type
    quant_types_in_specs = sorted({s.quant_type for s, _ in specs})
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

    for i, (spec, model_name) in enumerate(specs):
        key = _spec_key(spec, model_name)
        if key in completed:
            skipped += 1
            continue

        progress = f"[{i + 1}/{total}]"
        logger.info(
            "%s %s tokens=%d hidden=%d inter=%d experts=%d(local=%d) topk=%d quant=%s",
            progress, model_name, spec.num_tokens, spec.hidden_size,
            spec.intermediate_size, spec.num_experts, spec.local_num_experts,
            spec.topk, spec.quant_type,
        )

        try:
            moe_func = create_moe_func(spec, device)
            result = benchmark_npu(
                moe_func,
                warmup_iters=warmup_iters,
                num_runs=bench_iters,
            )
            row = _make_csv_row(spec, result, model_name)
            _, writer = csv_files[spec.quant_type]
            writer.writerow(row)

            graph_tag = "graph" if result.used_graph else "eager"
            logger.info(
                "%s -> %.2f us (avg of %d runs, %s)",
                progress, result.avg_us, result.num_runs, graph_tag,
            )
        except Exception:
            logger.exception(
                "%s FAILED %s tokens=%d quant=%s",
                progress, model_name, spec.num_tokens, spec.quant_type,
            )
            errors += 1
            continue
        finally:
            try:
                del moe_func
            except NameError:
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
        description="MoE microbenchmark collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./moe_data",
        help="Output directory for CSV files and checkpoint",
    )
    parser.add_argument(
        "--quant-types", nargs="+", default=["bf16"],
        choices=list(SUPPORTED_QUANT_TYPES),
        help="Quantization types to benchmark",
    )
    parser.add_argument(
        "--token-list", nargs="+", type=int, default=None,
        help="Token counts (default: built-in list)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names to benchmark (default: all). "
             f"Available: {[c[5] for c in MODEL_CONFIGS]}",
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

    token_list = args.token_list or DEFAULT_TOKEN_LIST

    # Filter model configs if --models specified
    if args.models:
        model_names = set(args.models)
        configs = [c for c in MODEL_CONFIGS if c[5] in model_names]
        unknown = model_names - {c[5] for c in configs}
        if unknown:
            logger.warning("Unknown model names (skipped): %s", unknown)
    else:
        configs = MODEL_CONFIGS

    specs = _build_spec_list(token_list, configs, args.quant_types)
    logger.info(
        "Parameter space: %d tokens x %d models x %d quant = %d total specs",
        len(token_list), len(configs), len(args.quant_types), len(specs),
    )

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
