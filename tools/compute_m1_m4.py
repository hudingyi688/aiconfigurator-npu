#!/usr/bin/env python3
"""Compute M1 / M5 coverage metrics for a GLM-5 inference run.

Analogous to MSMODELING's M1-M5 metrics, using per-op QuerySource
tags collected during inference. Run after source tracking was
added to PerformanceResult / base_backend (2026-06-05).

Metrics:
  M1  op-count hit rate  = SILICON ops / all non-ZERO ops
  M5  time-weighted coverage = (SILICON + PROFILER_DERIVED) latency / total latency
  (M2/M3 not implemented: require fused-group definitions)
  (M4 not implemented here: needs per-op shape info, see check_alignment.py)

Usage:
    python3 tools/compute_m1_m4.py
    python3 tools/compute_m1_m4.py --model zai-org/GLM-5 --phase prefill
    python3 tools/compute_m1_m4.py --phase decode
    python3 tools/compute_m1_m4.py --phase both
"""
from __future__ import annotations

import argparse
import importlib.resources as ir
from collections import Counter

from aiconfigurator_npu.sdk import common, config
from aiconfigurator_npu.sdk.backends.factory import get_backend
from aiconfigurator_npu.sdk.models import get_model
from aiconfigurator_npu.sdk.perf_database import get_database
from aiconfigurator_npu.sdk.performance_result import QuerySource

MODEL = "zai-org/GLM-5"
SYSTEM, BACKEND, VERSION = "ascend_910b", "vllm-ascend", "0.18.0"


def build_config(phase: str) -> config.ModelConfig:
    """Production GLM-5 parallel config for the given phase."""
    if phase == "prefill":
        return config.ModelConfig(
            tp_size=16, pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic,
            moe_quant_mode=common.MoEQuantMode.w8a8_dynamic,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            moe_tp_size=1, moe_ep_size=32, attention_dp_size=2,
            is_disagg_prefill=True,
        )
    else:  # decode
        return config.ModelConfig(
            tp_size=4, pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic,
            moe_quant_mode=common.MoEQuantMode.w8a8_dynamic,
            kvcache_quant_mode=common.KVCacheQuantMode.float16,
            fmha_quant_mode=common.FMHAQuantMode.float16,
            moe_tp_size=1, moe_ep_size=32, attention_dp_size=8,
        )


def compute_metrics(source_dict: dict, latency_dict: dict) -> dict:
    """Compute M1 and M5 from per-op source + latency dicts.

    Args:
        source_dict: {op_name: QuerySource | None}
        latency_dict: {op_name: latency_ms}

    Returns:
        dict with m1, m5, per-source counts and latency shares.
    """
    total_ops = len(source_dict)
    source_counts = Counter(v for v in source_dict.values())
    zero_count = source_counts.get(QuerySource.ZERO, 0)
    non_zero_ops = total_ops - zero_count

    silicon_ops = source_counts.get(QuerySource.SILICON, 0)
    profiler_ops = source_counts.get(QuerySource.PROFILER_DERIVED, 0)
    sol_ops = source_counts.get(QuerySource.SOL, 0)
    none_ops = source_counts.get(None, 0)

    m1 = silicon_ops / non_zero_ops if non_zero_ops > 0 else 0.0

    # M5: time-weighted coverage (SILICON + PROFILER_DERIVED latency / total)
    total_lat = sum(latency_dict.values())
    covered_lat = sum(
        lat for op, lat in latency_dict.items()
        if source_dict.get(op) in (QuerySource.SILICON, QuerySource.PROFILER_DERIVED)
    )
    m5 = covered_lat / total_lat if total_lat > 0 else 0.0

    return {
        "m1": m1,
        "m5": m5,
        "total_ops": total_ops,
        "non_zero_ops": non_zero_ops,
        "silicon_ops": silicon_ops,
        "profiler_derived_ops": profiler_ops,
        "sol_ops": sol_ops,
        "zero_ops": zero_count,
        "none_ops": none_ops,
        "total_latency_ms": total_lat,
        "covered_latency_ms": covered_lat,
    }


def print_report(phase: str, metrics: dict, source_dict: dict, latency_dict: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {phase.upper()} phase  —  M1 / M5 metrics")
    print(f"{'='*60}")
    print(f"  M1 (op-count hit rate):        {metrics['m1']:.1%}  "
          f"({metrics['silicon_ops']} SILICON / {metrics['non_zero_ops']} non-ZERO ops)")
    print(f"  M5 (time-weighted coverage):   {metrics['m5']:.1%}  "
          f"({metrics['covered_latency_ms']:.1f} / {metrics['total_latency_ms']:.1f} ms)")
    print(f"\n  Op breakdown:")
    print(f"    SILICON         : {metrics['silicon_ops']:>3}  (microbench real data)")
    print(f"    PROFILER_DERIVED: {metrics['profiler_derived_ops']:>3}  (profiler trace reverse-engineered)")
    print(f"    SOL             : {metrics['sol_ops']:>3}  (analytic roofline only)")
    print(f"    ZERO            : {metrics['zero_ops']:>3}  (gated off / pp=1 / agg KV)")
    print(f"    untagged        : {metrics['none_ops']:>3}")
    print(f"\n  Per-op detail (sorted by latency):")
    for op, lat in sorted(latency_dict.items(), key=lambda kv: -kv[1]):
        src = source_dict.get(op)
        src_str = src.value if src is not None else "?"
        pct = lat / metrics["total_latency_ms"] * 100 if metrics["total_latency_ms"] > 0 else 0
        print(f"    {op:<35}  {lat:>8.2f}ms  {pct:>5.1f}%  [{src_str}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--isl", type=int, default=10000)
    ap.add_argument("--osl", type=int, default=10)
    args = ap.parse_args()

    ROOT = str(ir.files("aiconfigurator_npu") / "systems")

    phases = ["prefill", "decode"] if args.phase == "both" else [args.phase]

    for phase in phases:
        mc = build_config(phase)
        m = get_model(args.model, mc, "vllm")
        db = get_database(SYSTEM, BACKEND, VERSION, ROOT)
        db.set_default_database_mode(common.DatabaseMode.HYBRID)
        b = get_backend("vllm")

        if phase == "prefill":
            rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=args.isl, osl=1)
            ctx_lat, _, gen_lat, _, ctx_src, gen_src = b._run_static_breakdown(
                m, db, rc, "static_ctx", 32, 1.0
            )
            lat_dict, src_dict = ctx_lat, ctx_src
        else:
            rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=4096, osl=args.osl)
            ctx_lat, _, gen_lat, _, ctx_src, gen_src = b._run_static_breakdown(
                m, db, rc, "static_gen", 1, 1.0
            )
            lat_dict, src_dict = gen_lat, gen_src

        metrics = compute_metrics(src_dict, lat_dict)
        print_report(phase, metrics, src_dict, lat_dict)


if __name__ == "__main__":
    main()
