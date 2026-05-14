#!/usr/bin/env bash
# Quick sanity check on the data/{gemm,moe,elementwise} csvs produced
# by tools/collect_no_attention.sh. Prints inventory, row counts,
# headers, sample rows, checkpoints, and a min/median/max summary
# of any latency-like column. Output stays under ~60 lines.
#
# Usage on the NPU host:
#     tools/inspect_no_attention_data.sh
#     tools/inspect_no_attention_data.sh /custom/data/root
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="${1:-data}"

echo "=== root: ${ROOT} ==="
ls -ld "${ROOT}" "${ROOT}"/gemm "${ROOT}"/moe "${ROOT}"/elementwise 2>/dev/null

echo ""
echo "=== file inventory (max 25 entries) ==="
find "${ROOT}/gemm" "${ROOT}/moe" "${ROOT}/elementwise" -maxdepth 2 \
     \( -name "*.csv" -o -name "checkpoint*" -o -name "*.json" \) \
     -printf "%-60p %10s bytes\n" 2>/dev/null | sort | head -25

echo ""
echo "=== row counts ==="
for f in "${ROOT}"/gemm/*.csv "${ROOT}"/moe/*.csv "${ROOT}"/elementwise/*.csv; do
  [ -f "$f" ] && printf "%-55s %10d rows\n" "$f" "$(wc -l < "$f")"
done

echo ""
echo "=== headers ==="
for f in "${ROOT}"/gemm/*.csv "${ROOT}"/moe/*.csv "${ROOT}"/elementwise/*.csv; do
  [ -f "$f" ] && { echo "-- $(basename "$f") --"; head -1 "$f"; }
done

echo ""
echo "=== first 2 data rows of each csv ==="
for f in "${ROOT}"/gemm/*.csv "${ROOT}"/moe/*.csv "${ROOT}"/elementwise/*.csv; do
  [ -f "$f" ] && { echo "-- $(basename "$f") --"; sed -n '2,3p' "$f"; }
done

echo ""
echo "=== latency / time stats per csv ==="
python3 - "${ROOT}" <<'PY'
import csv, glob, statistics, sys, os
root = sys.argv[1]
patterns = [f"{root}/gemm/*.csv", f"{root}/moe/*.csv", f"{root}/elementwise/*.csv"]
paths = sorted(p for pat in patterns for p in glob.glob(pat))
if not paths:
    print(f"(no csv files under {root})")
    raise SystemExit
for path in paths:
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"{os.path.basename(path)}: read failed: {e}")
        continue
    if not rows:
        print(f"{os.path.basename(path)}: empty (header only)")
        continue
    keys = list(rows[0].keys())
    cands = [
        c for c in keys
        if any(s in c.lower() for s in ("latency", "time_ms", "time_us",
                                        "duration", "elapsed", "_ms", "_us"))
    ]
    if not cands:
        print(f"{os.path.basename(path)}: n={len(rows)} cols={keys[:6]}{'...' if len(keys)>6 else ''} (no latency col)")
        continue
    col = cands[0]
    vals = []
    for r in rows:
        v = r.get(col)
        if v in (None, ""):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    if not vals:
        print(f"{os.path.basename(path)}: col {col!r} not numeric")
        continue
    nz = [v for v in vals if v > 0]
    print(f"{os.path.basename(path):28s} n={len(rows):5d}  "
          f"{col}: min={min(vals):.4g} med={statistics.median(vals):.4g} "
          f"max={max(vals):.4g} zeros={len(vals)-len(nz)}")
PY

echo ""
echo "=== run.log tail (last 8 lines) ==="
tail -8 "${ROOT}/run.log" 2>/dev/null || echo "(no run.log)"
