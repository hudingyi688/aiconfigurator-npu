"""Tests for QuerySource tracking and M1/M5 metric computation."""
from collections import Counter

import pytest

from aiconfigurator_npu.sdk.performance_result import PerformanceResult, QuerySource
from tools.compute_m1_m4 import compute_metrics


# ── QuerySource basics ────────────────────────────────────────────────────────

def test_query_source_values():
    assert QuerySource.SILICON.value == "SILICON"
    assert QuerySource.PROFILER_DERIVED.value == "PROFILER_DERIVED"
    assert QuerySource.SOL.value == "SOL"
    assert QuerySource.ZERO.value == "ZERO"


def test_performance_result_carries_source():
    r = PerformanceResult(10.5, energy=100.0, source=QuerySource.SILICON)
    assert r.source is QuerySource.SILICON
    assert float(r) == pytest.approx(10.5)
    assert r.energy == pytest.approx(100.0)


def test_performance_result_default_source_is_none():
    r = PerformanceResult(5.0, energy=0.0)
    assert r.source is None


def test_arithmetic_does_not_propagate_source():
    r1 = PerformanceResult(3.0, source=QuerySource.SILICON)
    r2 = PerformanceResult(2.0, source=QuerySource.SOL)
    total = r1 + r2
    assert total.source is None          # not propagated (plan §B)
    assert float(total) == pytest.approx(5.0)


def test_sum_does_not_propagate_source():
    results = [
        PerformanceResult(1.0, source=QuerySource.SILICON),
        PerformanceResult(2.0, source=QuerySource.PROFILER_DERIVED),
        PerformanceResult(3.0, source=QuerySource.SOL),
    ]
    total = sum(results)
    assert total.source is None
    assert float(total) == pytest.approx(6.0)


def test_scalar_multiply_clears_source():
    r = PerformanceResult(4.0, source=QuerySource.SILICON)
    scaled = r * 2
    assert scaled.source is None
    assert float(scaled) == pytest.approx(8.0)


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_m1_all_silicon():
    src = {"op1": QuerySource.SILICON, "op2": QuerySource.SILICON}
    lat = {"op1": 10.0, "op2": 5.0}
    m = compute_metrics(src, lat)
    assert m["m1"] == pytest.approx(1.0)
    assert m["m5"] == pytest.approx(1.0)


def test_m1_all_sol():
    src = {"op1": QuerySource.SOL, "op2": QuerySource.SOL}
    lat = {"op1": 10.0, "op2": 5.0}
    m = compute_metrics(src, lat)
    assert m["m1"] == pytest.approx(0.0)
    assert m["m5"] == pytest.approx(0.0)


def test_m1_excludes_zero_from_denominator():
    src = {
        "kv": QuerySource.PROFILER_DERIVED,   # covered
        "moe": QuerySource.SILICON,            # covered
        "gemm": QuerySource.SOL,               # not covered for M1
        "p2p": QuerySource.ZERO,               # excluded from M1 denom
    }
    lat = {"kv": 80.0, "moe": 10.0, "gemm": 5.0, "p2p": 0.0}
    m = compute_metrics(src, lat)
    # M1 = SILICON ops / non-ZERO ops = 1 / 3
    assert m["m1"] == pytest.approx(1 / 3)
    # M5 = (SILICON + PROFILER_DERIVED) lat / total = 90 / 95
    assert m["m5"] == pytest.approx(90 / 95)


def test_m5_profiler_derived_counts_as_covered():
    src = {"kv": QuerySource.PROFILER_DERIVED, "norm": QuerySource.SOL}
    lat = {"kv": 80.0, "norm": 20.0}
    m = compute_metrics(src, lat)
    assert m["m5"] == pytest.approx(0.8)


def test_glm5_decode_m1_m5_sanity():
    """Spot-check typical decode source breakdown (structural, not absolute values)."""
    # Reflects observed output from compute_m1_m4.py (tp4/ep32/dp8)
    src = {
        "generation_moe_overlap": QuerySource.SILICON,
        "generation_attention": QuerySource.SILICON,
        "generation_logits_gemm": QuerySource.SOL,
        "generation_add_norm_1": QuerySource.SOL,
        "generation_add_norm_2": QuerySource.SOL,
        "generation_embedding": QuerySource.SOL,
        "generation_p2p": QuerySource.ZERO,
    }
    lat = {
        "generation_moe_overlap": 236.0,
        "generation_attention": 214.0,
        "generation_logits_gemm": 2.7,
        "generation_add_norm_1": 2.1,
        "generation_add_norm_2": 2.1,
        "generation_embedding": 0.03,
        "generation_p2p": 0.0,
    }
    m = compute_metrics(src, lat)
    assert m["silicon_ops"] == 2
    assert m["zero_ops"] == 1
    assert m["non_zero_ops"] == 6
    assert m["m1"] == pytest.approx(2 / 6)
    # M5: SILICON covers moe+attention ≈ 450/457ms > 98%
    assert m["m5"] > 0.97
