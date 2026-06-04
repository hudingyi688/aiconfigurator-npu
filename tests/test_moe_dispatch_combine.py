"""Tests for the vllm-ascend fused dispatch+FFN+combine (FusedMC2) perf table.

Covers the loader (load_moe_dispatch_combine_data), the query method
(PerfDatabase.query_moe_dispatch_combine) including exact-grid lookup, linear
interpolation, and hold-flat clamping outside the measured range, plus the
operations-layer integration: on the vllm-ascend MoE-EP path the MoE FFN op
and the pre-dispatch comm op must drop out, with the fused silicon latency
emitted exactly once on the post-dispatch (combine) op.
"""

from __future__ import annotations

import importlib.resources as ir

import pytest

from aiconfigurator_npu.sdk import common
from aiconfigurator_npu.sdk import operations as ops
from aiconfigurator_npu.sdk.perf_database import (
    PerfDatabase,
    load_moe_dispatch_combine_data,
)

SYSTEM = "ascend_910b"
BACKEND = "vllm-ascend"
VERSION = "0.18.0"

# Known grid points (us in source CSV; loader converts us -> ms).
# ep2/bf16: num_tokens -> latency_us
EP2_BF16_256_US = 1827.9707
EP2_BF16_384_US = 1373.9777
EP2_BF16_1_US = 645.5292
EP2_BF16_4096_US = 1510.1260
EP8_W8A8_1_US = 302.9876


def _data_file() -> str:
    return str(
        ir.files("aiconfigurator_npu")
        / "systems"
        / "data"
        / SYSTEM
        / BACKEND
        / VERSION
        / "moe_dispatch_combine_perf.txt"
    )


@pytest.fixture(scope="module")
def raw_data():
    data = load_moe_dispatch_combine_data(_data_file())
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
    assert load_moe_dispatch_combine_data("/nonexistent/path/no_such_file.txt") is None


def test_loader_structure_and_grid(raw_data):
    assert sorted(raw_data.keys()) == [2, 4, 8, 32]
    for ep in (2, 4, 8):
        assert sorted(raw_data[ep].keys()) == ["bf16", "w8a8_dynamic"]
        # 24 distinct token counts per (ep, dtype)
        assert len(raw_data[ep]["bf16"]) == 24
        assert len(raw_data[ep]["w8a8_dynamic"]) == 24
    # ep32 is the production EP, collected on 2x A3 nodes (2x16 DIE). Its token
    # grid is the 13-point default sweep (sparser than the 24-point ep2/4/8
    # grids); interpolation over num_tokens covers the gaps.
    assert sorted(raw_data[32].keys()) == ["bf16", "w8a8_dynamic"]
    assert len(raw_data[32]["bf16"]) == 13
    assert len(raw_data[32]["w8a8_dynamic"]) == 13


def test_loader_unit_conversion_us_to_ms(raw_data):
    # 645.5292 us -> 0.6455292 ms
    assert raw_data[2]["bf16"][1]["latency"] == pytest.approx(EP2_BF16_1_US / 1000.0)
    assert raw_data[8]["w8a8_dynamic"][1]["latency"] == pytest.approx(EP8_W8A8_1_US / 1000.0)


def test_loader_leaf_shape(raw_data):
    leaf = raw_data[2]["bf16"][256]
    assert set(leaf.keys()) == {"latency", "power", "energy"}


# --------------------------------------------------------------------------- #
# Query: exact grid points
# --------------------------------------------------------------------------- #
def test_query_exact_grid_point(db):
    got = float(db.query_moe_dispatch_combine(256, "bf16", 2))
    assert got == pytest.approx(EP2_BF16_256_US / 1000.0)


def test_query_exact_grid_point_ep8(db):
    got = float(db.query_moe_dispatch_combine(1, "w8a8_dynamic", 8))
    assert got == pytest.approx(EP8_W8A8_1_US / 1000.0)


# --------------------------------------------------------------------------- #
# Query: interpolation
# --------------------------------------------------------------------------- #
def test_query_linear_interpolation(db):
    # 320 is midway between grid points 256 and 384
    got = float(db.query_moe_dispatch_combine(320, "bf16", 2))
    expected = (EP2_BF16_256_US + EP2_BF16_384_US) / 2.0 / 1000.0
    assert got == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------- #
# Query: clamping outside the measured range (hold flat, no extrapolation)
# --------------------------------------------------------------------------- #
def test_query_clamps_below_grid(db):
    # below smallest grid (1) -> held at the 1-token value
    got = float(db.query_moe_dispatch_combine(0, "bf16", 2))
    assert got == pytest.approx(EP2_BF16_1_US / 1000.0)


def test_query_clamps_above_grid(db):
    # far above largest grid (4096) -> held at the 4096-token value,
    # NOT linearly extrapolated to an absurd magnitude
    got = float(db.query_moe_dispatch_combine(10**6, "bf16", 2))
    assert got == pytest.approx(EP2_BF16_4096_US / 1000.0)


def test_query_at_exact_upper_boundary(db):
    got = float(db.query_moe_dispatch_combine(4096, "bf16", 2))
    assert got == pytest.approx(EP2_BF16_4096_US / 1000.0)


# --------------------------------------------------------------------------- #
# Query: error handling
# --------------------------------------------------------------------------- #
def test_query_unknown_dtype_raises(db):
    with pytest.raises(ValueError):
        db.query_moe_dispatch_combine(256, "fp8", 2)


def test_query_unmeasured_ep_snaps_to_nearest(db):
    # ep16 is not in the grid {2,4,8}; it must snap to the nearest measured ep
    # (8, held flat) rather than raising — a production ep16 prefill worker
    # must not crash the disagg search.
    got = float(db.query_moe_dispatch_combine(256, "bf16", 16))
    at_ep8 = float(db.query_moe_dispatch_combine(256, "bf16", 8))
    assert got == pytest.approx(at_ep8)


def test_query_ep_below_grid_snaps_to_smallest(db):
    # ep1 (below the smallest measured ep=2) holds flat at ep2
    got = float(db.query_moe_dispatch_combine(256, "bf16", 1))
    at_ep2 = float(db.query_moe_dispatch_combine(256, "bf16", 2))
    assert got == pytest.approx(at_ep2)



def test_query_latency_is_positive_across_grid(db):
    for ep in (2, 4, 8):
        for dtype in ("bf16", "w8a8_dynamic"):
            for ntok in (1, 64, 512, 4096):
                assert float(db.query_moe_dispatch_combine(ntok, dtype, ep)) > 0.0


# --------------------------------------------------------------------------- #
# Integration: MoE triplet collapses to a single fused number on vllm-ascend
# --------------------------------------------------------------------------- #
# Model dims matching the silicon table (hidden=6144, inter=2048, topk=8,
# num_experts=256). moe_tp=1, attention_dp=1 -> the vllm-ascend MoE-EP path.
_H, _INTER, _TOPK, _NE = 6144, 2048, 8, 256
_TP, _DP = 1, 1


def _triplet(ep, quant_mode):
    """Build (pre_dispatch, moe_ffn, post_dispatch) ops for the MoE-EP path."""
    pre = ops.MoEDispatch(
        "pre", 1, _H, _TOPK, _NE, _TP, ep, _DP, True, quant_mode=quant_mode, is_context=True
    )
    moe = ops.MoE(
        "moe", 1, _H, _INTER, _TOPK, _NE, _TP, ep, quant_mode, "power_law_1.2", _DP, is_context=True
    )
    post = ops.MoEDispatch(
        "post", 1, _H, _TOPK, _NE, _TP, ep, _DP, False, quant_mode=quant_mode, is_context=True
    )
    return pre, moe, post


@pytest.mark.parametrize(
    "quant_mode,dtype_tag",
    [
        (common.MoEQuantMode.w8a8_dynamic, "w8a8_dynamic"),
        (common.MoEQuantMode.float16, "bf16"),
    ],
)
@pytest.mark.parametrize("ep", [2, 4, 8])
def test_moe_ffn_op_is_zeroed_on_ep_path(db, ep, quant_mode, dtype_tag):
    # The expert FFN compute is folded into the fused silicon number, so the
    # paired MoE op must contribute nothing on this gated path.
    _pre, moe, _post = _triplet(ep, quant_mode)
    assert float(moe.query(db, x=256)) == 0.0


@pytest.mark.parametrize(
    "quant_mode,dtype_tag",
    [
        (common.MoEQuantMode.w8a8_dynamic, "w8a8_dynamic"),
        (common.MoEQuantMode.float16, "bf16"),
    ],
)
@pytest.mark.parametrize("ep", [2, 4, 8])
def test_fused_silicon_counted_once_on_combine_op(db, ep, quant_mode, dtype_tag):
    # The fused dispatch+FFN+combine latency must appear exactly once, on the
    # post-dispatch (combine) op. The attention-side allreduce (ar_latency) is
    # a separate, pre-existing kernel charged identically on both pre and post
    # ops, so it cancels in (post - pre), isolating the fused number.
    pre, _moe, post = _triplet(ep, quant_mode)
    fused = float(db.query_moe_dispatch_combine(256, dtype_tag, ep))
    r_pre = float(pre.query(db, x=256))
    r_post = float(post.query(db, x=256))
    assert (r_post - r_pre) == pytest.approx(fused, abs=1e-6)


def test_pre_dispatch_carries_no_fused_silicon(db):
    # Pre-dispatch must NOT add the fused number (only the combine op does).
    # With attention_tp == ep > 1, pre still carries ar_latency, so compare
    # against a bare allreduce rather than zero.
    pre, _moe, _post = _triplet(4, common.MoEQuantMode.w8a8_dynamic)
    fused = float(db.query_moe_dispatch_combine(256, "w8a8_dynamic", 4))
    r_pre = float(pre.query(db, x=256))
    # pre is far smaller than the fused kernel — it carries only ar_latency
    assert r_pre < fused


# --------------------------------------------------------------------------- #
# Integration: DP attention (attention_dp_size > 1) still takes the fused path
# --------------------------------------------------------------------------- #
# vllm-ascend picks FusedMC2 on ep_world_size alone, regardless of DP attention,
# so the mainstream DP-attn + EP-MoE form (dp>1) must behave identically: the
# MoE FFN op zeroes out, the fused silicon lands once on combine, and the
# AllGather dp_latency path is suppressed (FusedMC2 does no cross-DP all_gather,
# so charging dp_latency on top would double-count). Pick dp == ep (moe_tp=1)
# so attention_tp_size == 1 and the attention-side ar_latency drops out too,
# isolating the fused number cleanly.
def _triplet_dp(ep, dp, quant_mode):
    """Build (pre, moe, post) ops for the DP-attention MoE-EP path."""
    pre = ops.MoEDispatch(
        "pre", 1, _H, _TOPK, _NE, _TP, ep, dp, True, quant_mode=quant_mode, is_context=True
    )
    moe = ops.MoE(
        "moe", 1, _H, _INTER, _TOPK, _NE, _TP, ep, quant_mode, "power_law_1.2", dp, is_context=True
    )
    post = ops.MoEDispatch(
        "post", 1, _H, _TOPK, _NE, _TP, ep, dp, False, quant_mode=quant_mode, is_context=True
    )
    return pre, moe, post


@pytest.mark.parametrize("ep", [2, 4, 8])
def test_moe_ffn_op_is_zeroed_with_dp_attention(db, ep):
    # dp == ep > 1: MoE FFN op must still drop out (fused number owns the FFN).
    _pre, moe, _post = _triplet_dp(ep, ep, common.MoEQuantMode.w8a8_dynamic)
    assert float(moe.query(db, x=256)) == 0.0


@pytest.mark.parametrize("ep", [2, 4, 8])
def test_fused_path_suppresses_dp_latency(db, ep):
    # dp == ep > 1 -> attention_tp_size == 1, so neither ar_latency nor (on the
    # fused path) dp_latency is charged. The combine op must equal exactly the
    # fused silicon number, and the pre op must be ~0 — proving the AllGather
    # dp_latency branch did NOT also fire (which would inflate both ops).
    pre, _moe, post = _triplet_dp(ep, ep, common.MoEQuantMode.w8a8_dynamic)
    fused = float(db.query_moe_dispatch_combine(256, "w8a8_dynamic", ep))
    r_pre = float(pre.query(db, x=256))
    r_post = float(post.query(db, x=256))
    assert r_pre == pytest.approx(0.0, abs=1e-9)
    assert r_post == pytest.approx(fused, abs=1e-6)


