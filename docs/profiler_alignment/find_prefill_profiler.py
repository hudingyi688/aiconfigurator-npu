#!/usr/bin/env python3
"""Scan a directory tree for Ascend `kernel_details.csv` profiler files and
report which ones are usable for **prefill DSA absolute-value validation**.

A profiler is usable only if BOTH hold (see docs/profiler_alignment notes):
  1. SFA num_heads == per-rank heads of the target TP (GLM-5: 64 heads / tp16 = 4).
     A mismatch (e.g. 64 = TP1 full heads) makes SFA compute differ ~16x — not comparable.
  2. The window is a COMPLETE request, not a sampled fragment:
     sum(SFA query dim) / num_layers ≈ isl. A fragment (e.g. 626 tok/layer << 10000)
     cannot validate the per-step DSA accumulation.

Usage:
    python3 find_prefill_profiler.py [ROOT] [--heads 4] [--layers 80] \
        [--isl 10000] [--isl-tol 0.2]

ROOT defaults to the current directory. Exit code is 0 if at least one usable
profiler is found, 1 otherwise (handy for scripting on the NPU box).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter
from dataclasses import dataclass

SFA_PREFIX = "SparseFlashAttention"
KERNEL_GLOB = "kernel_details.csv"


@dataclass(frozen=True)
class ProbeResult:
    """Immutable summary of one kernel_details.csv probe."""

    path: str
    num_heads: int | None
    sfa_count: int
    query_per_layer: float
    error: str | None = None

    def verdict(self, want_heads: int, isl: int, isl_tol: float) -> str:
        if self.error:
            return "ERR"
        if self.sfa_count == 0:
            return "SKIP"  # no DSA attention — not a GLM-5/DSA run
        heads_ok = self.num_heads == want_heads
        lo, hi = isl * (1 - isl_tol), isl * (1 + isl_tol)
        isl_ok = lo <= self.query_per_layer <= hi
        return "OK_USABLE" if (heads_ok and isl_ok) else "NO"


def probe(path: str, num_layers: int) -> ProbeResult:
    """Read one kernel_details.csv and extract SFA num_heads + query coverage.

    Validates at the boundary: malformed shapes are skipped, not trusted.
    Returns an immutable ProbeResult; never raises (errors are captured).
    """
    query_sum = 0
    heads_counter: Counter[int] = Counter()
    sfa_count = 0
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if not row.get("Name", "").startswith(SFA_PREFIX):
                    continue
                shape = row.get("Input Shapes", "").strip().strip('"')
                first = shape.split(";")[0].split(",")
                if len(first) < 2:
                    continue
                try:
                    query_dim, heads_dim = int(first[0]), int(first[1])
                except ValueError:
                    continue  # non-numeric shape token — skip, don't trust
                query_sum += query_dim
                heads_counter[heads_dim] += 1
                sfa_count += 1
    except (OSError, csv.Error) as exc:
        return ProbeResult(path, None, 0, 0.0, error=str(exc))

    num_heads = heads_counter.most_common(1)[0][0] if heads_counter else None
    query_per_layer = query_sum / num_layers if num_layers else 0.0
    return ProbeResult(path, num_heads, sfa_count, query_per_layer)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find prefill profilers usable for DSA absolute-value validation."
    )
    p.add_argument("root", nargs="?", default=".", help="root dir to scan (default: .)")
    p.add_argument("--heads", type=int, default=4,
                   help="required per-rank SFA heads (GLM-5 tp16 = 64/16 = 4)")
    p.add_argument("--layers", type=int, default=80, help="model layers (GLM-5 = 80)")
    p.add_argument("--isl", type=int, default=10000, help="target input seq len")
    p.add_argument("--isl-tol", type=float, default=0.2,
                   help="fractional tolerance on query/layer vs isl (default 0.2)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    pattern = os.path.join(args.root, "**", KERNEL_GLOB)
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No {KERNEL_GLOB} found under {args.root!r}", file=sys.stderr)
        return 1

    usable: list[ProbeResult] = []
    for path in files:
        res = probe(path, args.layers)
        verdict = res.verdict(args.heads, args.isl, args.isl_tol)
        if verdict == "SKIP":
            continue  # quiet: not a DSA run
        if verdict == "ERR":
            print(f"ERR  {path}: {res.error}")
            continue
        nh = res.num_heads
        reason = ""
        if verdict == "NO":
            bad_h = "" if nh == args.heads else f" heads≠{args.heads}"
            lo, hi = args.isl * (1 - args.isl_tol), args.isl * (1 + args.isl_tol)
            bad_i = "" if lo <= res.query_per_layer <= hi else " fragment(q/layer≪isl)"
            reason = f" <-{bad_h}{bad_i}"
        print(f"{verdict:9} nh={nh} q/layer={res.query_per_layer:.0f} "
              f"SFA={res.sfa_count}  {path}{reason}")
        if verdict == "OK_USABLE":
            usable.append(res)

    if usable:
        print(f"\n{len(usable)} usable profiler(s). Use this one:")
        print(f"  {usable[0].path}")
        return 0
    print("\nNo usable profiler. Re-collect: P-node, tp16(nh4), "
          "--max-concurrency 1 --num-prompts 3 --random-input-len 10000.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
