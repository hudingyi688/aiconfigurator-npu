# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PerformanceResult class for backward-compatible latency+energy tracking.
"""

from enum import Enum


class QuerySource(Enum):
    """Data source classification for a perf-database query result.

    Used to compute M1-M5 coverage metrics (analogous to MSMODELING's QuerySource).
    Source is per-op only; it is NOT propagated through arithmetic operations —
    use the per-op source_dict from InferenceSummary for metric computation.
    """
    SILICON = "SILICON"              # bench table hit or interpolation (real measured data)
    PROFILER_DERIVED = "PROFILER_DERIVED"  # profiler trace reverse-engineered (KV transfer, prefill DSA)
    SOL = "SOL"                      # analytic roofline only, no bench data
    ZERO = "ZERO"                    # zero-cost op (gated off, pp=1 P2P, agg KV transfer, etc.)


class PerformanceResult(float):
    """
    Float-like class that stores latency, energy, and data-source classification.

    Behaves exactly like a float for backward compatibility, but stores energy
    and an optional QuerySource tag internally.

    Source semantics: meaningful only on the direct return value of a query_*
    call. Arithmetic operations (sum, +, *, /) do NOT propagate source — the
    result carries source=None. Use the per-op source_dict from InferenceSummary
    for M1-M5 metric computation.

    Units:
        - latency: milliseconds (ms)
        - energy: watt-milliseconds (W·ms) = millijoules (mJ)
        - power: watts (W) - derived property
    """

    # Note: We don't use __slots__ here because float subclasses cannot define __slots__

    def __new__(cls, latency, energy=0.0, source=None):
        instance = float.__new__(cls, latency)
        return instance

    def __init__(self, latency, energy=0.0, source=None):
        self.energy = energy  # W·ms (watt-milliseconds)
        self.source: QuerySource | None = source

    @property
    def power(self):
        latency = float(self)
        if latency > 1e-9:
            return self.energy / latency
        return 0.0

    def __repr__(self):
        return (
            f"PerformanceResult(latency={float(self)}, energy={self.energy}, "
            f"power={self.power}, source={self.source})"
        )

    # Arithmetic: source is NOT propagated (per-op only semantic, plan §B)
    def __add__(self, other):
        if isinstance(other, PerformanceResult):
            return PerformanceResult(float(self) + float(other), energy=self.energy + other.energy)
        return PerformanceResult(float(self) + other, energy=self.energy)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        return PerformanceResult(float(self) * other, energy=self.energy * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return PerformanceResult(float(self) / other, energy=self.energy / other)

    def __rtruediv__(self, other):
        return other / float(self)

    # Comparison operators (CRITICAL - Python doesn't auto-infer from float inheritance)
    def __lt__(self, other):
        return float(self) < float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __eq__(self, other):
        try:
            return float(self) == float(other)
        except (TypeError, ValueError):
            return False

    def __ne__(self, other):
        try:
            return float(self) != float(other)
        except (TypeError, ValueError):
            return True

    def __abs__(self):
        return PerformanceResult(abs(float(self)), energy=abs(self.energy))

    def __hash__(self):
        return hash((float(self), self.energy))

    def __str__(self):
        return str(float(self))

