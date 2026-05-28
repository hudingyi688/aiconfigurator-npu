#!/usr/bin/env python3
"""Build comm_calibration.json from Ascend profiler runs.

Each (op_kind, ep_size) factor is anchored to its OWN SOL alpha-beta
prediction — NOT divided through ep=1. This avoids the prior
methodology bug where ep=1 (degenerate dispatch, no real alltoallv
traffic) was used as the baseline, making ep=8 look 5x faster than
the model and ep=16 look 2x slower than the model.

Inputs:
    Directory tree like
        glm-profiler/dp{D}_pp{P}_tp{T}_dcp{C}_ep{E}_rank0_{phase}-{tokens}/
            ASCEND_PROFILER_OUTPUT/kernel_details.csv

Output:
    docs/profiler_alignment/comm_calibration.json with schema
        {"factors": {"<op_kind>@ep<N>": float, ...}, "doc": "..."}

For each run we take the median Duration(us) per kernel family. We
then median across runs sharing the same ep_size. The SOL reference
is computed at a fixed (H=6144, M=128, K=8) — the GLM-5 production
"typical" decode point — using aiconfigurator's own query_nccl(SOL)
so the ratio is consistent with what the search engine sees.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


# ---- kernel name → op_kind classification --------------------------
def classify(name: str) -> str | None:
    """Return one of our op_kind labels or None if not a comm op."""
    n = name.lower()
    if n.startswith("dispatchffncombine"):
        return "moe_dispatch_combine_w8a8"
    if "moedistributedispatchv" in n:
        return "moe_dispatch_bf16"
    if "moedistributecombinev" in n:
        return "moe_combine_bf16"
    if n.startswith("hcom_allreduce"):
        return "all_reduce"
    if n.startswith("hcom_allgather"):
        return "all_gather"
    if n.startswith("hcom_reducescatter"):
        return "reduce_scatter"
    if n.startswith("hcom_alltoallv") or n.startswith("hcom_alltoall"):
        return "all_to_all"
    return None


_RUN_RE = re.compile(r"dp(\d+)_pp(\d+)_tp(\d+)_dcp(\d+)_ep(\d+)_rank")


def parse_ep(run_dir_name: str) -> int | None:
    m = _RUN_RE.match(run_dir_name)
    if not m:
        return None
    return int(m.group(5))


# ---- profiler aggregation ------------------------------------------
def median_durations_per_kind(kernel_csv: Path) -> dict[str, float]:
    """Return median Duration(us) per op_kind in this run."""
    durations: dict[str, list[float]] = defaultdict(list)
    with kernel_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kind = classify(row["Name"])
            if kind is None:
                continue
            try:
                durations[kind].append(float(row["Duration(us)"]))
            except (ValueError, KeyError):
                continue
    return {k: median(v) for k, v in durations.items() if v}


def aggregate_runs(profiler_root: Path) -> dict[tuple[str, int], float]:
    """Median-of-medians per (op_kind, ep_size). Skips ep=0 baseline."""
    per_run: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run_dir in sorted(profiler_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ep = parse_ep(run_dir.name)
        if ep is None or ep == 0:
            continue
        kernel_csv = run_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        if not kernel_csv.is_file():
            continue
        meds = median_durations_per_kind(kernel_csv)
        for op_kind, lat_us in meds.items():
            per_run[(op_kind, ep)].append(lat_us)
    return {key: median(v) for key, v in per_run.items()}


# ---- SOL reference latency from aiconfigurator ---------------------
def sol_latency_us(database, op_kind: str, ep_size: int) -> float:
    """SOL alpha-beta latency in microseconds for the reference shape.

    Reference: GLM-5 H=6144, M=128 (per-rank tokens), topk=8.
    For dispatch+combine ops the volume is volume*topk (full alltoallv
    payload per rank). For the bare collectives we use volume.
    """
    from aiconfigurator_npu.sdk import common

    H = 6144
    M = 128
    K = 8
    volume = M * H

    op_to_collective = {
        "moe_dispatch_combine_w8a8": ("all_to_all", volume * K),
        "moe_dispatch_bf16": ("all_to_all", volume * K),
        "moe_combine_bf16": ("all_to_all", volume * K),
        "all_reduce": ("all_reduce", volume),
        "all_gather": ("all_gather", volume),
        "reduce_scatter": ("reduce_scatter", volume),
        "all_to_all": ("all_to_all", volume),
    }
    if op_kind not in op_to_collective:
        raise KeyError(op_kind)
    collective, msg_size = op_to_collective[op_kind]

    result = database.query_nccl(
        common.CommQuantMode.half,
        ep_size,
        collective,
        msg_size,
        database_mode=common.DatabaseMode.SOL,
    )
    # query_nccl returns ms; convert to us for parity with profiler
    return float(result) * 1000.0


def build_database():
    import importlib.resources as p

    from aiconfigurator_npu.sdk import common
    from aiconfigurator_npu.sdk.perf_database import PerfDatabase

    sr = str(p.files("aiconfigurator_npu") / "systems")
    return PerfDatabase(
        "ascend_910b", common.BackendName.vllm_ascend.value, "0.18.0", systems_root=sr
    )


# ---- main ----------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: build_comm_calibration.py <profiler_root> <output_json>",
            file=sys.stderr,
        )
        sys.exit(2)

    profiler_root = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    profiler_med = aggregate_runs(profiler_root)
    if not profiler_med:
        print(f"no profiler data found under {profiler_root}", file=sys.stderr)
        sys.exit(1)

    db = build_database()

    factors: dict[str, float] = {}
    debug_rows: list[tuple[str, int, float, float, float]] = []
    for (op_kind, ep), prof_us in sorted(profiler_med.items()):
        try:
            sol_us = sol_latency_us(db, op_kind, ep)
        except Exception as e:
            print(f"  skip {op_kind}@ep{ep}: SOL lookup failed ({e})", file=sys.stderr)
            continue
        if sol_us <= 0:
            print(f"  skip {op_kind}@ep{ep}: SOL=0", file=sys.stderr)
            continue
        factor = prof_us / sol_us
        factors[f"{op_kind}@ep{ep}"] = round(factor, 4)
        debug_rows.append((op_kind, ep, prof_us, sol_us, factor))

    # Print a small audit table to stderr
    print(f"{'op_kind':30s} {'ep':>3s} {'prof_us':>10s} {'sol_us':>10s} {'factor':>8s}",
          file=sys.stderr)
    for op_kind, ep, prof_us, sol_us, factor in debug_rows:
        print(f"{op_kind:30s} {ep:3d} {prof_us:10.3f} {sol_us:10.3f} {factor:8.4f}",
              file=sys.stderr)

    doc = (
        "Per (op_kind, ep_size) correction factor: profiler median latency / "
        "aiconfigurator SOL alpha-beta latency at GLM-5 reference shape (H=6144, "
        "M=128, K=8). Each ep is anchored to its OWN SOL — the previous version "
        "divided through ep=1, which gave misleadingly small factors at intra-node "
        "ep and inflated factors at cross-node ep. Apply as scalar multiplier on "
        "top of aiconfigurator's analytical comm model."
    )
    payload = {"doc": doc, "reference_shape": {"H": 6144, "M": 128, "K": 8},
               "factors": factors}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {len(factors)} entries to {output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
