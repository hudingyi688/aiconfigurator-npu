#!/usr/bin/env python3
"""Convert TensorCast HCCL csvs into aiconfigurator perf-table format.

Source (real measurements from production HCCL bench):
    .../hccl/v8.5/hcom_allGather_.csv
    .../hccl/v8.5/hcom_allReduce_.csv
    .../hccl/v8.5/hcom_alltoallv_.csv
    .../hccl/v8.5/hcom_reduceScatter_.csv

Each source row:
    message_bytes, num_devices, dtype, topology_tier,
    Duration(us), bandwidth_gbps

Output (overwrite):
    systems/data/ascend_910b/vllm-ascend/0.18.0/
                     ├── nccl_perf.txt
                     │     nccl_dtype, num_gpus, message_size,
                     │     kernel_source, op_name, latency (ms)
                     └── custom_allreduce_perf.txt
                           framework, version, device, op_name,
                           kernel_source, allreduce_dtype, num_gpus,
                           message_size, latency (ms), backend

Mapping rules:
    DT_BF16 / DT_FLOAT16 / DT_FP16  ->  half
    DT_FLOAT / DT_FLOAT32           ->  float
    DT_INT8                         ->  int8

    Duration(us) / 1000             ->  latency (ms)

Notes:
    AllReduce data is written to BOTH files because:
      - aiconfigurator vllm-ascend AllReduce path goes through
        query_custom_allreduce() (operations.py:718)
      - cross-node fallback (tp > num_gpus_per_node) inside
        query_custom_allreduce dispatches to query_nccl

    AllGather / ReduceScatter / AllToAll only go through
    query_nccl, no custom_allreduce equivalent.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convert_hccl")


# Map TensorCast dtype tags -> aiconfigurator CommQuantMode names.
DTYPE_MAP = {
    "DT_BF16": "half",
    "DT_FLOAT16": "half",
    "DT_FP16": "half",
    "DT_HALF": "half",
    "DT_FLOAT": "float",
    "DT_FLOAT32": "float",
    "DT_INT8": "int8",
}

# Map source filename -> aiconfigurator op_name.
OP_MAP = {
    "hcom_allGather_.csv":     "all_gather",
    "hcom_allReduce_.csv":     "all_reduce",
    "hcom_alltoallv_.csv":     "all_to_all",
    "hcom_reduceScatter_.csv": "reduce_scatter",
}


def _read_hccl(src: Path, op_name: str) -> list[dict]:
    """Read one HCCL csv, return list of normalized rows."""
    rows = []
    skipped = 0
    with src.open() as f:
        for r in csv.DictReader(f):
            dtype = DTYPE_MAP.get(r["dtype"].strip())
            if dtype is None:
                skipped += 1
                continue
            try:
                msg = int(r["message_bytes"])
                ng = int(r["num_devices"])
                lat_ms = float(r["Duration(us)"]) / 1000.0
            except (KeyError, ValueError):
                skipped += 1
                continue
            rows.append(
                {"op_name": op_name, "nccl_dtype": dtype,
                 "num_gpus": ng, "message_size": msg, "latency": lat_ms}
            )
    log.info("  %-28s %5d rows (%d skipped)", src.name, len(rows), skipped)
    return rows


def _write_nccl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["nccl_dtype", "num_gpus", "message_size",
             "kernel_source", "op_name", "latency"]
        )
        for r in rows:
            w.writerow([r["nccl_dtype"], r["num_gpus"], r["message_size"],
                        "HCCL", r["op_name"], f"{r['latency']:.6g}"])
    log.info("nccl_perf.txt: wrote %d rows -> %s", len(rows), out)


def _write_custom_allreduce(
    rows: list[dict],
    out: Path,
    framework: str,
    version: str,
    device: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "framework", "version", "device",
            "op_name", "kernel_source",
            "allreduce_dtype", "num_gpus", "message_size",
            "latency", "backend",
        ])
        for r in rows:
            # The "AUTO" backend label matches what perf_database.query_custom_allreduce
            # selects when looking up by allreduce strategy.
            w.writerow([
                framework, version, device,
                r["op_name"], "HCCL",
                r["nccl_dtype"], r["num_gpus"], r["message_size"],
                f"{r['latency']:.6g}", "AUTO",
            ])
    log.info("custom_allreduce_perf.txt: wrote %d rows -> %s", len(rows), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing the four hcom_*.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="systems/data/<device>/<framework>/<version>/ target directory",
    )
    parser.add_argument("--framework", default="vllm-ascend")
    parser.add_argument("--version", default="0.18.0")
    parser.add_argument("--device", default="ascend_910b")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"input dir not found: {args.input_dir}")

    log.info("reading HCCL csvs from %s", args.input_dir)

    all_rows: list[dict] = []
    allreduce_rows: list[dict] = []
    for fname, op_name in OP_MAP.items():
        src = args.input_dir / fname
        if not src.exists():
            log.warning("missing: %s", src)
            continue
        rows = _read_hccl(src, op_name)
        all_rows.extend(rows)
        if op_name == "all_reduce":
            allreduce_rows.extend(rows)

    if not all_rows:
        raise SystemExit("no rows loaded; aborting")

    _write_nccl(all_rows, args.output_dir / "nccl_perf.txt")
    _write_custom_allreduce(
        allreduce_rows,
        args.output_dir / "custom_allreduce_perf.txt",
        framework=args.framework,
        version=args.version,
        device=args.device,
    )

    # Summary
    by_op = {}
    for r in all_rows:
        by_op.setdefault(r["op_name"], 0)
        by_op[r["op_name"]] += 1
    log.info("done: %d total rows", len(all_rows))
    for op, c in sorted(by_op.items()):
        log.info("  %-15s %5d rows", op, c)


if __name__ == "__main__":
    main()
