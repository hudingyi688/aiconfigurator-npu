"""NPU Event-based timing engine for operator benchmarking.

Aligned with AIConfigurator's benchmark_with_power() timing rules:
- NPU Graph capture + replay to eliminate Python dispatch overhead
- Single Event pair wrapping multiple runs (no per-iteration synchronize)
- latency = total_elapsed / num_runs / repeat_n
- Fallback to eager execution if graph capture fails
- Dual-mode: runs both graph and eager, takes the minimum
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchResult:
    """Immutable benchmark result."""

    avg_us: float
    num_runs: int
    repeat_n: int
    used_graph: bool = False
    graph_us: float = 0.0
    eager_us: float = 0.0


def _timed_run(
    kernel_func: Callable[[], None],
    num_runs: int,
    repeat_n: int,
    graph=None,
) -> float:
    """Run kernel_func num_runs times and return average latency in us.

    If graph is provided, uses graph.replay() instead of calling kernel_func.
    """
    start_evt = torch.npu.Event(enable_timing=True)
    end_evt = torch.npu.Event(enable_timing=True)

    start_evt.record()
    for _ in range(num_runs):
        if graph is not None:
            graph.replay()
        else:
            for _ in range(repeat_n):
                kernel_func()
    end_evt.record()
    torch.npu.synchronize()

    return start_evt.elapsed_time(end_evt) * 1000.0 / num_runs / repeat_n
