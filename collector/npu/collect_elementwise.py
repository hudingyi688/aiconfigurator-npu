"""ElementWise operator microbenchmark collector for Ascend NPU.

Covers small but non-negligible operators via vllm-ascend / torch_npu APIs,
ensuring the same CANN kernel path as production inference:
  - RmsNorm: torch_npu.npu_rms_norm()
  - RoPE: torch_npu._npu_rotary_embedding()
  - SwiGLU: torch_npu.npu_swiglu()
  - Softmax: torch.nn.functional.softmax()

Usage:
    python collect_elementwise.py --hidden-list 2048 5120 7168 \\
        --batch-list 1 16 --output-dir ./elementwise_data
"""

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElemSpec:
    op_type: str
    batch: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_size: int
    dtype: torch.dtype = torch.bfloat16


# ── Operator factories (aligned with vllm-ascend kernel path) ──


def _create_rmsnorm(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """RmsNorm via torch_npu.npu_rms_norm — same as vllm_ascend/ops/layernorm.py:82.

    Independent RmsNorm without residual add. Used for final_layernorm and QK norm.
    """
    import torch_npu  # noqa: F401

    weight = torch.ones(spec.hidden_size, dtype=spec.dtype, device=device)
    x = torch.randn(spec.batch, spec.hidden_size, dtype=spec.dtype, device=device)
    eps = 1e-6

    def forward():
        torch_npu.npu_rms_norm(x, weight, eps)

    forward()
    torch.npu.synchronize()
    return forward


def _create_add_rmsnorm(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """Fused Add+RmsNorm via torch_npu.npu_add_rms_norm — same as vllm_ascend/ops/layernorm.py:77.

    Fused residual add + RmsNorm. This is the dominant path (~97.7% of RmsNorm calls),
    used for input_layernorm and post_attention_layernorm in every layer.
    """
    import torch_npu  # noqa: F401

    weight = torch.ones(spec.hidden_size, dtype=spec.dtype, device=device)
    x = torch.randn(spec.batch, spec.hidden_size, dtype=spec.dtype, device=device)
    residual = torch.randn(spec.batch, spec.hidden_size, dtype=spec.dtype, device=device)
    eps = 1e-6

    def forward():
        torch_npu.npu_add_rms_norm(x, residual, weight, eps)

    forward()
    torch.npu.synchronize()
    return forward


def _create_rope(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """RoPE via torch_npu._npu_rotary_embedding — same as vllm_ascend/ops/rotary_embedding.py:189."""
    import torch_npu  # noqa: F401

    num_tokens = spec.batch
    num_heads = spec.num_heads
    head_size = spec.head_size
    rotary_dim = head_size  # full rotary

    q = torch.randn(num_tokens, num_heads * head_size, dtype=spec.dtype, device=device)
    k = torch.randn(num_tokens, num_heads * head_size, dtype=spec.dtype, device=device)
    cos_sin_cache = torch.randn(32768, rotary_dim, dtype=spec.dtype, device=device)
    positions = torch.arange(num_tokens, device=device)

    def forward():
        torch_npu._npu_rotary_embedding(
            positions, q, k, head_size, cos_sin_cache, True,
        )

    forward()
    torch.npu.synchronize()
    return forward


def _create_swiglu(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """SwiGLU via torch_npu.npu_swiglu — same as vllm_ascend/ops/activation.py:38."""
    import torch_npu  # noqa: F401

    x = torch.randn(
        spec.batch, spec.intermediate_size * 2, dtype=spec.dtype, device=device,
    )

    def forward():
        torch_npu.npu_swiglu(x)

    forward()
    torch.npu.synchronize()
    return forward


def _create_softmax(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """Softmax over vocab dimension — LM head sampling softmax.

    In production, this is called once per decode step with shape (1, vocab_size).
    The hidden_size parameter is repurposed as vocab_size here.
    """
    x = torch.randn(1, spec.hidden_size, dtype=spec.dtype, device=device)

    def forward():
        torch.nn.functional.softmax(x, dim=-1)

    forward()
    torch.npu.synchronize()
    return forward


OP_FACTORIES = {
    "rmsnorm": _create_rmsnorm,
    "add_rmsnorm": _create_add_rmsnorm,
    "rope": _create_rope,
    "swiglu": _create_swiglu,
    "softmax": _create_softmax,
}


def main():
    parser = argparse.ArgumentParser(description="ElementWise operator microbenchmark")
    parser.add_argument(
        "--op-types", nargs="+", default=list(OP_FACTORIES.keys()),
        choices=list(OP_FACTORIES.keys()),
    )
    parser.add_argument("--hidden-list", nargs="+", type=int, default=[2048, 4096, 5120, 7168, 8192])
    parser.add_argument("--intermediate-list", nargs="+", type=int, default=[1024, 2048, 3200, 4096])
    parser.add_argument("--vocab-size-list", nargs="+", type=int, default=[32000, 128256, 151936, 152064])
    parser.add_argument("--batch-list", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--num-heads-list", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./elementwise_data")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    device = torch.device("npu")

    os.makedirs(args.output_dir, exist_ok=True)

    from bench_engine import benchmark_npu

    fieldnames = [
        "Op Type", "Batch", "Hidden Size", "Intermediate Size",
        "Num Heads", "Head Size", "Average Duration(us)",
    ]

    csv_path = os.path.join(args.output_dir, "elementwise_perf.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    total = 0
    errors = 0

    for op_type in args.op_types:
        factory = OP_FACTORIES[op_type]

        if op_type == "rope":
            specs = [
                ElemSpec(op_type, b, h, 0, nh, args.head_size)
                for b in args.batch_list
                for h in args.hidden_list
                for nh in args.num_heads_list
            ]
        elif op_type == "swiglu":
            specs = [
                ElemSpec(op_type, b, h, inter, 0, args.head_size)
                for b in args.batch_list
                for h in args.hidden_list
                for inter in args.intermediate_list
            ]
        elif op_type == "softmax":
            # Softmax uses vocab_size as hidden_size dimension
            specs = [
                ElemSpec(op_type, 1, vocab, 0, 0, args.head_size)
                for vocab in args.vocab_size_list
            ]
        elif op_type in ("rmsnorm", "add_rmsnorm"):
            specs = [
                ElemSpec(op_type, b, h, 0, 0, args.head_size)
                for b in args.batch_list
                for h in args.hidden_list
            ]
        else:
            specs = [
                ElemSpec(op_type, b, h, 0, 0, args.head_size)
                for b in args.batch_list
                for h in args.hidden_list
            ]

        for spec in specs:
            total += 1
            try:
                func = factory(spec, device)
                result = benchmark_npu(
                    func,
                    warmup_iters=args.warmup_iters,
                    num_runs=args.bench_iters,
                    repeat_n=6,
                )
                writer.writerow({
                    "Op Type": op_type,
                    "Batch": spec.batch,
                    "Hidden Size": spec.hidden_size,
                    "Intermediate Size": spec.intermediate_size,
                    "Num Heads": spec.num_heads,
                    "Head Size": spec.head_size,
                    "Average Duration(us)": f"{result.avg_us:.2f}",
                })
                graph_tag = "graph" if result.used_graph else "eager"
                logger.info(
                    "%s batch=%d hidden=%d -> %.2f us (%s)",
                    op_type, spec.batch, spec.hidden_size, result.avg_us, graph_tag,
                )
            except Exception:
                logger.exception(
                    "FAILED %s batch=%d hidden=%d",
                    op_type, spec.batch, spec.hidden_size,
                )
                errors += 1
            finally:
                torch.npu.empty_cache()

    f.close()
    logger.info(
        "Done: %d benchmarked, %d errors. Output: %s",
        total - errors, errors, csv_path,
    )


if __name__ == "__main__":
    main()
