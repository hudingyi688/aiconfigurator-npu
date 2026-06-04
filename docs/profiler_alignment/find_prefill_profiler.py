#!/usr/bin/env python3
"""Scan a directory tree for Ascend `kernel_details.csv` profiler files and
report which ones are usable for **prefill DSA absolute-value validation**.

GLM-5 DSA prefill uses Context Parallelism (CP): each rank keeps ALL heads
(nh=64) but processes only 1/cp of the query tokens (sequence-dim split via
all-to-all + o_proj full-gather). This is DIFFERENT from decode, which is
tensor-parallel head-split (tp4 -> nh16). So for a prefill profiler:

  1. SFA num_heads should equal the FULL head count (GLM-5: 64), regardless of
     tp. A head-split value (e.g. 4) would mean a non-CP / non-prefill capture.
  2. The window is a COMPLETE request when sum(SFA query)/layers ≈ isl/cp
     (per-rank query after CP split), NOT ≈ isl. e.g. 10k / cp16 ≈ 626 tok/layer
     is COMPLETE, not a fragment — the earlier "fragment" reading was wrong.

Usage:
    python3 find_prefill_profiler.py [ROOT] [--heads 64] [--layers 80] \
        [--isl 10000] [--cp 16] [--isl-tol 0.2]

The completeness target is isl/cp. Set --cp 1 to require the full isl per rank
(non-CP capture). ROOT defaults to the current directory. Exit code is 0 if at
least one usable profiler is found, 1 otherwise (handy for scripting on NPU).
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

    def per_rank_target(self, isl: int, cp: int) -> float:
        """Expected per-rank query/layer after CP split = isl / cp."""
        return isl / cp if cp else float(isl)

    def verdict(self, want_heads: int, isl: int, cp: int, isl_tol: float) -> str:
        if self.error:
            return "ERR"
        if self.sfa_count == 0:
            return "SKIP"  # no DSA attention — not a GLM-5/DSA run
        heads_ok = self.num_heads == want_heads
        target = self.per_rank_target(isl, cp)
        lo, hi = target * (1 - isl_tol), target * (1 + isl_tol)
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
    p.add_argument("--heads", type=int, default=64,
                   help="required SFA heads (GLM-5 DSA prefill = full 64, CP keeps all heads)")
    p.add_argument("--layers", type=int, default=80, help="model layers (GLM-5 = 80)")
    p.add_argument("--isl", type=int, default=10000, help="target GLOBAL input seq len")
    p.add_argument("--cp", type=int, default=16,
                   help="context-parallel size; per-rank query target = isl/cp (default 16 = tp16)")
    p.add_argument("--isl-tol", type=float, default=0.2,
                   help="fractional tolerance on query/layer vs isl/cp (default 0.2)")
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
        verdict = res.verdict(args.heads, args.isl, args.cp, args.isl_tol)
        if verdict == "SKIP":
            continue  # quiet: not a DSA run
        if verdict == "ERR":
            print(f"ERR  {path}: {res.error}")
            continue
        nh = res.num_heads
        target = res.per_rank_target(args.isl, args.cp)
        reason = ""
        if verdict == "NO":
            bad_h = "" if nh == args.heads else f" heads≠{args.heads}"
            lo, hi = target * (1 - args.isl_tol), target * (1 + args.isl_tol)
            bad_i = "" if lo <= res.query_per_layer <= hi else \
                f" coverage≠isl/cp(want≈{target:.0f})"
            reason = f" <-{bad_h}{bad_i}"
        print(f"{verdict:9} nh={nh} q/layer={res.query_per_layer:.0f} "
              f"(target≈{target:.0f}) SFA={res.sfa_count}  {path}{reason}")
        if verdict == "OK_USABLE":
            usable.append(res)

    if usable:
        print(f"\n{len(usable)} usable profiler(s). Use this one:")
        print(f"  {usable[0].path}")
        return 0
    print("\nNo usable profiler. Re-collect: P-node, full prefill, "
          f"--max-concurrency 1 --num-prompts 3 --random-input-len {args.isl}. "
          f"Expect nh={args.heads}, q/layer≈{args.isl/args.cp:.0f} (CP{args.cp}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
