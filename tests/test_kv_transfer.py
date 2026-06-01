"""Tests for the vllm-ascend PD-disaggregated KV-transfer cost model.

Covers the loader (load_kv_transfer_data), the query method
(PerfDatabase.query_kv_transfer) including exact-grid lookup, isl/ep
interpolation, both-axis hold-flat clamping (stored value is the measured net KV wall-clock, no overlap_factor); plus the
operations-layer KVTransfer gating: the cost is charged ONCE on the vllm-ascend
disagg prefill path and is zero everywhere else (agg mode, decode, missing
table).

Emphasis is on boundaries and the gate (per project "0 happy path"): clamps,
unknown keys, the agg/disagg gate, and the missing-arg contract — not just the
nominal grid lookup.
"""

from __future__ import annotations

import importlib.resources as ir

import pytest

from aiconfigurator_npu.sdk import common
from aiconfigurator_npu.sdk import operations as ops
from aiconfigurator_npu.sdk.perf_database import (
    PerfDatabase,
    load_kv_transfer_data,
)

SYSTEM = "ascend_910b"
BACKEND = "vllm-ascend"
VERSION = "0.18.0"

# Known grid points (net_kv_wallclock_ms in source file).
EP1_2500 = 867.0
EP1_10000 = 1260.4
EP1_20000 = 1785.1
EP16_2500 = 1011.0
EP16_10000 = 1679.4
EP16_20000 = 2571.0


def _data_file() -> str:
    return str(
        ir.files("aiconfigurator_npu")
        / "systems"
        / "data"
        / SYSTEM
        / BACKEND
        / VERSION
        / "kv_transfer_perf.txt"
    )


@pytest.fixture(scope="module")
def raw_data():
    data = load_kv_transfer_data(_data_file())
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
    assert load_kv_transfer_data("/nonexistent/path/no_such_file.txt") is None


def test_loader_structure_and_grid(raw_data):
    assert sorted(raw_data.keys()) == [1, 16]
    for ep in (1, 16):
        assert sorted(raw_data[ep].keys()) == [2500, 10000, 20000]


def test_loader_values(raw_data):
    assert raw_data[1][2500] == pytest.approx(EP1_2500)
    assert raw_data[16][20000] == pytest.approx(EP16_20000)


# --------------------------------------------------------------------------- #
# Query: exact grid points (returns stored net KV wall-clock directly)
# --------------------------------------------------------------------------- #
def test_query_exact_grid_ep1(db):
    got = float(db.query_kv_transfer(2500, 1))
    assert got == pytest.approx(EP1_2500)


def test_query_exact_grid_ep16(db):
    got = float(db.query_kv_transfer(20000, 16))
    assert got == pytest.approx(EP16_20000)


# --------------------------------------------------------------------------- #
# Query: interpolation on each axis
# --------------------------------------------------------------------------- #
def test_query_isl_interpolation(db):
    # 15000 is midway between 10000 and 20000 on the ep16 row
    got = float(db.query_kv_transfer(15000, 16))
    expected = (EP16_10000 + EP16_20000) / 2.0
    assert got == pytest.approx(expected, abs=1e-3)


def test_query_ep_interpolation(db):
    # ep=8 lies between ep1 and ep16 at fraction (8-1)/(16-1) on isl=10000
    got = float(db.query_kv_transfer(10000, 8))
    frac = (8 - 1) / (16 - 1)
    expected = EP1_10000 + frac * (EP16_10000 - EP1_10000)
    assert got == pytest.approx(expected, abs=1e-2)


# --------------------------------------------------------------------------- #
# Query: both-axis clamping (hold flat, no extrapolation)
# --------------------------------------------------------------------------- #
def test_query_clamps_isl_below(db):
    # isl below smallest grid (2500) -> held at the 2500 value
    got = float(db.query_kv_transfer(1, 16))
    assert got == pytest.approx(EP16_2500)


def test_query_clamps_isl_above(db):
    got = float(db.query_kv_transfer(10**6, 1))
    assert got == pytest.approx(EP1_20000)


def test_query_clamps_ep_below(db):
    # ep below smallest grid (1) clamps to the ep=1 (dp-only) row
    got = float(db.query_kv_transfer(10000, 0))
    assert got == pytest.approx(EP1_10000)


def test_query_clamps_ep_above(db):
    got = float(db.query_kv_transfer(10000, 64))
    assert got == pytest.approx(EP16_10000)


def test_query_monotonic_in_isl_at_ep16(db):
    a = float(db.query_kv_transfer(2500, 16))
    b = float(db.query_kv_transfer(10000, 16))
    c = float(db.query_kv_transfer(20000, 16))
    assert a < b < c


# --------------------------------------------------------------------------- #
# Operations layer: KVTransfer gate
# --------------------------------------------------------------------------- #
def test_op_charges_cost_on_disagg_prefill(db):
    op = ops.KVTransfer("kv", 1, 16, is_disagg_prefill=True)
    got = float(op.query(db, s=20000))
    assert got == pytest.approx(EP16_20000)


def test_op_zero_in_agg_mode(db):
    # is_disagg_prefill=False -> no cross-worker KV transfer to charge
    op = ops.KVTransfer("kv", 1, 16, is_disagg_prefill=False)
    assert float(op.query(db, s=20000)) == 0.0


def test_op_scale_factor_applied(db):
    op = ops.KVTransfer("kv", 2.0, 16, is_disagg_prefill=True)
    got = float(op.query(db, s=20000))
    assert got == pytest.approx(EP16_20000 * 2.0)


def test_op_ep1_path(db):
    op = ops.KVTransfer("kv", 1, 1, is_disagg_prefill=True)
    got = float(op.query(db, s=2500))
    assert got == pytest.approx(EP1_2500)


def test_op_missing_isl_raises(db):
    # s (isl) is required; int(None) must fail loudly rather than silently 0
    op = ops.KVTransfer("kv", 1, 16, is_disagg_prefill=True)
    with pytest.raises(TypeError):
        op.query(db)


def test_op_energy_is_zero(db):
    op = ops.KVTransfer("kv", 1, 16, is_disagg_prefill=True)
    result = op.query(db, s=10000)
    assert getattr(result, "energy", 0.0) == 0.0


