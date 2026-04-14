"""ElementWise operator microbenchmark collector for Ascend NPU.

Covers small but non-negligible operators that are not GEMM/Attention/MoE:
  - RoPE (Rotary Position Embedding)
  - RmsNorm (Root Mean Square Normalization)
  - SwiGLU (SiLU + Gated Linear Unit activation)
  - Softmax

These operators are memory-bound and shape-fixed per model, but on NPU
their latency includes significant CANN dispatch overhead that cannot be
estimated from memory bandwidth alone. Bench data is needed for accurate
E2E prediction.

Usage:
    python collect_elementwise.py --hidden-list 2048 5120 7168 \
        --batch-list 1 16 --output-dir ./elementwise_data
"""

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
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


# ── Operator factories ──

def _create_rmsnorm(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """RmsNorm: input (batch, hidden) → normalized (batch, hidden)."""
    weight = torch.ones(spec.hidden_size, dtype=spec.dtype, device=device)
    x = torch.randn(spec.batch, spec.hidden_size, dtype=spec.dtype, device=device)
    variance_epsilon = 1e-6

    def forward():
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden = x * torch.rsqrt(variance + variance_epsilon)
        return hidden * weight

    forward()
    torch.npu.synchronize()
    return forward


def _create_rope(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """RoPE: apply rotary embedding to Q/K tensors."""
    seq_len = spec.batch  # reuse batch as seq_len for RoPE
    q = torch.randn(seq_len, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)
    k = torch.randn(seq_len, spec.num_heads, spec.head_size, dtype=spec.dtype, device=device)

    # Pre-compute cos/sin tables
    half_dim = spec.head_size // 2
    freqs = torch.randn(seq_len, half_dim, dtype=spec.dtype, device=device)
    cos = torch.cos(freqs).unsqueeze(1)
    sin = torch.sin(freqs).unsqueeze(1)

    def forward():
        q1, q2 = q[..., :half_dim], q[..., half_dim:]
        k1, k2 = k[..., :half_dim], k[..., half_dim:]
        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_rot, k_rot

    forward()
    torch.npu.synchronize()
    return forward


def _create_swiglu(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """SwiGLU: gate_up (batch, inter*2) → silu(gate) * up → (batch, inter)."""
    x = torch.randn(spec.batch, spec.intermediate_size * 2, dtype=spec.dtype, device=device)

    def forward():
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    forward()
    torch.npu.synchronize()
    return forward


def _create_softmax(spec: ElemSpec, device: torch.device) -> Callable[[], None]:
    """Softmax over last dimension."""
    x = torch.randn(spec.batch, spec.hidden_size, dtype=spec.dtype, device=device)

    def forward():
        return torch.nn.functional.softmax(x, dim=-1)

    forward()
    torch.npu.synchronize()
    return forward


OP_FACTORIES = {
    "rmsnorm": _create_rmsnorm,
    "rope": _create_rope,
    "swiglu": _create_swiglu,
    "softmax": _create_softmax,
}


def main():
    parser = argparse.ArgumentParser(description="ElementWise operator microbenchmark")
    parser.add_argument("--op-types", nargs="+", default=list(OP_FACTORIES.keys()),
                        choices=list(OP_FACTORIES.keys()))
    parser.add_argument("--hidden-list", nargs="+", type=int, default=[2048, 4096, 5120, 7168, 8192])
    parser.add_argument("--intermediate-list", nargs="+", type=int, default=[1024, 2048, 3200, 4096])
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

    fieldnames = ["Op Type", "Batch", "Hidden Size", "Intermediate Size",
                  "Num Heads", "Head Size", "Average Duration(us)"]

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
        else:  # rmsnorm, softmax
            specs = [
                ElemSpec(op_type, b, h, 0, 0, args.head_size)
                for b in args.batch_list
                for h in args.hidden_list
            ]

        for spec in specs:
            total += 1
            try:
                func = factory(spec, device)
                result = benchmark_npu(func, warmup_iters=args.warmup_iters,
                                       num_runs=args.bench_iters, repeat_n=6)
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
                logger.info("%s batch=%d hidden=%d -> %.2f us (%s)",
                            op_type, spec.batch, spec.hidden_size, result.avg_us, graph_tag)
            except Exception:
                logger.exception("FAILED %s batch=%d hidden=%d", op_type, spec.batch, spec.hidden_size)
                errors += 1
            finally:
                torch.npu.empty_cache()

    f.close()
    logger.info("Done: %d benchmarked, %d errors. Output: %s", total - errors, errors, csv_path)


if __name__ == "__main__":
    main()
