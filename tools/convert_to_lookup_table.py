#!/usr/bin/env python3
"""Convert NPU-collected csvs (data/{gemm,moe}/) into the upstream
aiconfigurator perf-table format under
    systems/data/<device>/<framework>/<version>/.

Mirrors the existing layout under
    systems/data/ascend_910b/vllm-ascend/0.18.0/
which uses these schemas:

  gemm_perf.txt:
    framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency

  moe_perf.txt:
    framework,version,device,op_name,kernel_source,moe_dtype,
    num_tokens,hidden_size,inter_size,topk,num_experts,
    moe_tp_size,moe_ep_size,distribution,latency

`latency` is in milliseconds (collector csvs record us, so we divide
by 1000).

Attention is NOT generated: this NPU box's OPP is missing the
prebuilt FIA / SFA binaries for head_dim=192 — see
docs/MISSING_DATA.md.

Usage:
    python tools/convert_to_lookup_table.py
        [--input-root data]
        [--output-root systems/data]
        [--device ascend_910_93]
        [--framework vllm-ascend]
        [--version 0.18.0]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convert")


# Map collector quant-type tags -> GEMMQuantMode/MoEQuantMode tag names
# in src/aiconfigurator_npu/sdk/common.py. The aiconfigurator perf
# loader compares the perf.txt's *dtype column to these enum names,
# so values must match exactly:
#   GEMMQuantMode.float16        -> "float16"   (BF16 / FP16)
#   GEMMQuantMode.w8a8_dynamic   -> "w8a8_dynamic"
DTYPE_MAP = {
    "bf16": "float16",
    "w8a8_dynamic": "w8a8_dynamic",
}

US_TO_MS = 1.0 / 1000.0


def _parse_gemm_shape(shape_str: str) -> tuple[int, int, int] | None:
    """Parse the collector's "Input Shapes" cell, e.g.
        "1,256;256,256"      -> m=1, k=256, n=256
        "1,512;256,512"      -> m=1, k=512, n=256

    The convention in collect_gemm.py is x:(m,k), weight:(n,k), so
    the second tensor's first dim is n and second dim is k.
    """
    parts = shape_str.split(";")
    if len(parts) != 2:
        return None
    a = [int(x) for x in parts[0].split(",")]
    b = [int(x) for x in parts[1].split(",")]
    if len(a) != 2 or len(b) != 2:
        return None
    m, k1 = a
    n, k2 = b
    if k1 != k2:
        return None
    return m, n, k1


def _open_writer(path: Path, header: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.writer(f)
    w.writerow(header)
    return f, w


def convert_gemm(
    input_root: Path,
    output: Path,
    framework: str,
    version: str,
    device: str,
) -> int:
    header = ["framework", "version", "device", "op_name", "kernel_source",
              "gemm_dtype", "m", "n", "k", "latency"]
    src_files = sorted(input_root.glob("gemm/*.csv"))
    if not src_files:
        log.warning("no gemm csvs under %s", input_root / "gemm")
        return 0
    f, w = _open_writer(output, header)
    n_rows = 0
    n_skipped = 0
    try:
        for src in src_files:
            with src.open() as inp:
                for row in csv.DictReader(inp):
                    quant = (row.get("Quant Type") or "").strip()
                    dtype = DTYPE_MAP.get(quant)
                    if dtype is None:
                        n_skipped += 1
                        continue
                    parsed = _parse_gemm_shape(row.get("Input Shapes", ""))
                    if parsed is None:
                        n_skipped += 1
                        continue
                    m, n, k = parsed
                    try:
                        lat_us = float(row["Average Duration(us)"])
                    except (KeyError, ValueError):
                        n_skipped += 1
                        continue
                    w.writerow([
                        framework, version, device,
                        "gemm", "vllm_ascend_default",
                        dtype, m, n, k,
                        f"{lat_us * US_TO_MS:.6g}",
                    ])
                    n_rows += 1
    finally:
        f.close()
    log.info("gemm: wrote %d rows to %s (skipped %d)", n_rows, output, n_skipped)
    return n_rows


_QUANT_FROM_FNAME = re.compile(r"_(BF16|W8A8)\.csv$", re.IGNORECASE)


def convert_moe(
    input_root: Path,
    output: Path,
    framework: str,
    version: str,
    device: str,
) -> int:
    header = ["framework", "version", "device", "op_name", "kernel_source",
              "moe_dtype",
              "num_tokens", "hidden_size", "inter_size", "topk", "num_experts",
              "moe_tp_size", "moe_ep_size", "distribution", "latency"]
    src_files = sorted(input_root.glob("moe/*.csv"))
    if not src_files:
        log.warning("no moe csvs under %s", input_root / "moe")
        return 0
    f, w = _open_writer(output, header)
    n_rows = 0
    n_skipped = 0
    try:
        for src in src_files:
            with src.open() as inp:
                for row in csv.DictReader(inp):
                    quant = (row.get("Quant Type") or "").strip()
                    dtype = DTYPE_MAP.get(quant)
                    if dtype is None:
                        n_skipped += 1
                        continue
                    try:
                        num_tokens = int(row["Num Tokens"])
                        hidden     = int(row["Hidden Size"])
                        inter      = int(row["Intermediate Size"])
                        experts    = int(row["Num Experts"])
                        topk       = int(row["TopK"])
                        ep_size    = int(row.get("EP Size") or 1)
                        lat_us     = float(row["Average Duration(us)"])
                    except (KeyError, ValueError):
                        n_skipped += 1
                        continue
                    w.writerow([
                        framework, version, device,
                        "moe", "vllm_ascend_fused_moe",
                        dtype,
                        num_tokens, hidden, inter, topk, experts,
                        1, ep_size, "power_law_1.2",
                        f"{lat_us * US_TO_MS:.6g}",
                    ])
                    n_rows += 1
    finally:
        f.close()
    log.info("moe: wrote %d rows to %s (skipped %d)", n_rows, output, n_skipped)
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="data", type=Path)
    ap.add_argument("--output-root", default="systems/data", type=Path)
    ap.add_argument("--device", default="ascend_910_93")
    ap.add_argument("--framework", default="vllm-ascend")
    ap.add_argument("--version", default="0.18.0")
    args = ap.parse_args()

    out_dir = args.output_root / args.device / args.framework / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gemm = convert_gemm(
        args.input_root, out_dir / "gemm_perf.txt",
        args.framework, args.version, args.device,
    )
    n_moe = convert_moe(
        args.input_root, out_dir / "moe_perf.txt",
        args.framework, args.version, args.device,
    )

    log.info("done: %d gemm rows + %d moe rows -> %s", n_gemm, n_moe, out_dir)
    log.info("attention not generated (see docs/MISSING_DATA.md)")


if __name__ == "__main__":
    main()
