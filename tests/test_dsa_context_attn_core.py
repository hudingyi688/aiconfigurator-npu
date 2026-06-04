"""Tests for the profiler-derived DSA prefill attention-core cost model.

Covers the loader (load_dsa_context_attn_core_data), the per-layer query
(query_dsa_context_attn_core) including exact-grid lookup, linear interpolation,
both-end hold-flat clamping (SFA saturates at the high end), the projection-GEMM
SOL complement (query_context_dsa_projection_sol), and the ContextDSAModule
two-regime behaviour: profiler path when the table is loaded (with trailing
partial-chunk query scaling) vs the legacy whole-module path otherwise.

Emphasis is on boundaries and contracts (per project "0 happy path"): clamps,
partial last chunk, the table-loaded gate, and cross-validation against the
production profiler baselines (10k attention-core 152.5ms, 20k 392.3ms).
"""

from __future__ import annotations

import importlib.resources as ir
import math

import pytest

from aiconfigurator_npu.sdk import common
from aiconfigurator_npu.sdk import operations as ops
from aiconfigurator_npu.sdk.perf_database import (
    PerfDatabase,
    load_dsa_context_attn_core_data,
)

SYSTEM = "ascend_910b"
BACKEND = "vllm-ascend"
VERSION = "0.18.0"

# Known grid points (per_layer_us in source file) -> ms.
GRID_US = {4096: 290, 8192: 1037, 12288: 1105, 16384: 1196, 20480: 1278}


def _data_file() -> str:
    return str(
        ir.files("aiconfigurator_npu")
        / "systems" / "data" / SYSTEM / BACKEND / VERSION
        / "dsa_context_attn_core_perf.txt"
    )


@pytest.fixture(scope="module")
def raw_data():
    data = load_dsa_context_attn_core_data(_data_file())
    assert data is not None, "data file should exist and load"
    return data


@pytest.fixture(scope="module")
def db():
    root = str(ir.files("aiconfigurator_npu") / "systems")
    return PerfDatabase(SYSTEM, BACKEND, VERSION, root)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_loader_returns_none_for_missing_file():
    assert load_dsa_context_attn_core_data("/nonexistent/no_such_file.txt") is None


def test_loader_structure_and_us_to_ms(raw_data):
    assert sorted(raw_data.keys()) == sorted(GRID_US.keys())
    for kv, us in GRID_US.items():
        assert raw_data[kv] == pytest.approx(us / 1000.0)


# --------------------------------------------------------------------------- #
# query_dsa_context_attn_core — exact, interp, clamp
# --------------------------------------------------------------------------- #
def test_query_exact_grid_points(db):
    for kv, us in GRID_US.items():
        assert float(db.query_dsa_context_attn_core(cum_kv=kv)) == pytest.approx(us / 1000.0)


def test_query_linear_interpolation_between_points(db):
    # midpoint of 4096 (290us) and 8192 (1037us) -> ~663.5us
    mid = float(db.query_dsa_context_attn_core(cum_kv=6144)) * 1000
    assert mid == pytest.approx((290 + 1037) / 2, rel=0.01)


def test_query_clamps_below_low_end(db):
    # below 4096 holds flat at the 4096 value (chunked prefill never goes lower)
    assert float(db.query_dsa_context_attn_core(cum_kv=1)) == pytest.approx(290 / 1000.0)
    assert float(db.query_dsa_context_attn_core(cum_kv=2000)) == pytest.approx(290 / 1000.0)


def test_query_clamps_above_high_end_sfa_saturated(db):
    # above 20480 holds flat (SFA saturated at the sparse topk cap)
    top = 1278 / 1000.0
    assert float(db.query_dsa_context_attn_core(cum_kv=32768)) == pytest.approx(top)
    assert float(db.query_dsa_context_attn_core(cum_kv=200000)) == pytest.approx(top)


# --------------------------------------------------------------------------- #
# query_context_dsa_projection_sol — scales with query, not KV
# --------------------------------------------------------------------------- #
def test_projection_scales_linearly_with_query(db):
    a = float(db.query_context_dsa_projection_sol(
        q=128, num_heads=64, gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic))
    b = float(db.query_context_dsa_projection_sol(
        q=256, num_heads=64, gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic))
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_projection_independent_of_cum_kv(db):
    # projection takes no KV arg at all — it touches only the new query tokens
    v = float(db.query_context_dsa_projection_sol(
        q=256, num_heads=64, gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic))
    assert v > 0.0


# --------------------------------------------------------------------------- #
# ContextDSAModule — profiler path, partial last chunk, cross-validation
# --------------------------------------------------------------------------- #
def _module(cp_size=16):
    return ops.ContextDSAModule(
        "context_attention", 80, 64,
        common.KVCacheQuantMode.float16, common.FMHAQuantMode.float16,
        common.GEMMQuantMode.w8a8_dynamic,
        architecture="DeepseekV32ForCausalLM", cp_size=cp_size,
    )


def test_module_full_chunks_match_profiler_20k(db):
    # 20k actually pads to 20480 = 5 full 4096-chunks; attention-core baseline
    # (measured) = 392.3ms. Subtract the analytical projection to compare cores.
    m = _module()
    total = float(m.query(db, batch_size=1, s=20480, prefix=0))
    proj = float(db.query_context_dsa_projection_sol(
        q=256, num_heads=64, gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic))
    proj_total = proj * 80 * 5  # 5 full chunks
    core = total - proj_total
    assert core == pytest.approx(392.3, rel=0.05)


def test_module_partial_last_chunk_scales_down(db):
    # 10k: last chunk processes only 1808 global tokens (q_frac=1808/4096), so
    # the model must NOT charge a full 3rd chunk. Core lands near 152.5ms.
    m = _module()
    total = float(m.query(db, batch_size=1, s=10000, prefix=0))
    proj = float(db.query_context_dsa_projection_sol(
        q=256, num_heads=64, gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic))
    step_kvs = [min((k + 1) * 4096, 10000) for k in range(3)]
    proj_total = sum(proj * 80 * max(0.0, min(1.0, (kv - k * 4096) / 4096))
                     for k, kv in enumerate(step_kvs))
    core = total - proj_total
    assert core == pytest.approx(152.5, rel=0.10)


def test_module_partial_chunk_cheaper_than_full(db):
    # An isl whose last chunk is half-full must cost less than rounding it up to
    # a full extra chunk (guards against the pre-fix over-charge).
    m = _module()
    half = float(m.query(db, batch_size=1, s=4096 + 2048, prefix=0))
    full = float(m.query(db, batch_size=1, s=4096 + 4096, prefix=0))
    assert half < full


def test_module_falls_back_when_table_absent(monkeypatch, db):
    # When the profiler table is not loaded, the module must use the legacy
    # whole-module path (query_context_dsa_module), not crash.
    m = _module()
    called = {}

    class _Stub:
        loaded = False

    monkeypatch.setattr(db, "_dsa_context_attn_core_data", _Stub())

    def _fake_whole_module(**kwargs):
        called["hit"] = True
        from aiconfigurator_npu.sdk.perf_database import PerformanceResult
        return PerformanceResult(0.123, energy=0.0)

    monkeypatch.setattr(db, "query_context_dsa_module", _fake_whole_module)
    out = float(m.query(db, batch_size=1, s=8192, prefix=0))
    assert called.get("hit") is True
    assert out > 0.0

