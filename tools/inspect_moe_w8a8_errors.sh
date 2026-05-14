#!/usr/bin/env bash
# Investigate the 26 errors reported by collect_moe.py during the
# W8A8 sweep. The previous greps for "FAILED" came up empty, so this
# script casts a wider net and also inspects the moe checkpoint to
# tell us how many W8A8 specs actually landed.
#
# Usage on the NPU host:
#     tools/inspect_moe_w8a8_errors.sh
#     tools/inspect_moe_w8a8_errors.sh /custom/data/root
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="${1:-data}"
LOG="${ROOT}/run_w8a8.log"

echo "=== ${LOG} basic stats ==="
if [[ ! -f "${LOG}" ]]; then
  echo "(missing) ${LOG}"
  exit 1
fi
wc -l "${LOG}"
echo "size: $(du -h "${LOG}" | cut -f1)"

echo ""
echo "=== summary lines (Done: ... benchmarked / skipped / errors) ==="
grep -E "Done:|benchmarked|skipped|errors" "${LOG}" | head -10

echo ""
echo "=== any error/exception/fail/traceback line (case-insensitive) ==="
# Cap at 30 so output stays bounded.
grep -niE "error|traceback|exception|fail|raise|except |out of memory|oom" "${LOG}" \
  | head -30

echo ""
echo "=== last 100 lines of log ==="
tail -100 "${LOG}"

echo ""
echo "=== moe checkpoint analysis ==="
python3 - "${ROOT}" <<'PY'
import json, os, sys, collections
root = sys.argv[1]
ckpt = os.path.join(root, "moe", "moe_checkpoint.json")
if not os.path.exists(ckpt):
    print(f"(missing) {ckpt}")
    raise SystemExit
with open(ckpt) as f:
    data = json.load(f)
completed = list(data.get("completed", []))
print(f"checkpoint total : {len(completed)}")
buckets = collections.Counter()
for k in completed:
    if   "w8a8" in k.lower(): buckets["w8a8"] += 1
    elif "bf16" in k.lower(): buckets["bf16"] += 1
    else:                     buckets["other"] += 1
for kind, n in sorted(buckets.items()):
    print(f"  {kind:6s} entries: {n}")
print("first 4 w8a8 keys :")
for k in [x for x in completed if "w8a8" in x.lower()][:4]: print(" ", k)
print("first 4 bf16 keys :")
for k in [x for x in completed if "bf16" in x.lower()][:4]: print(" ", k)
PY

echo ""
echo "=== csv row counts (what actually made it to disk) ==="
for f in "${ROOT}"/moe/*.csv; do
  [ -f "$f" ] && printf "%-55s %10d rows\n" "$f" "$(wc -l < "$f")"
done

echo ""
echo "=== reconcile: spec count vs csv vs checkpoint ==="
python3 - "${ROOT}" <<'PY'
import csv, glob, json, os, sys
root = sys.argv[1]
ckpt = os.path.join(root, "moe", "moe_checkpoint.json")
ck_w8a8 = 0
if os.path.exists(ckpt):
    with open(ckpt) as f:
        ck_w8a8 = sum(1 for k in json.load(f).get("completed", []) if "w8a8" in k.lower())

w8a8_csv = os.path.join(root, "moe", "GroupedMatmul_MoE_W8A8.csv")
csv_rows = 0
if os.path.exists(w8a8_csv):
    with open(w8a8_csv) as f:
        csv_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
print(f"w8a8 csv rows           : {csv_rows}")
print(f"w8a8 checkpoint entries : {ck_w8a8}")
print(f"delta (csv - ckpt)      : {csv_rows - ck_w8a8}")
print("Interpretation:")
print(" - if delta == 0 and ckpt < csv+errors: errors didn't write to ckpt")
print(" - if csv == ckpt and run_w8a8 reports 26 errors:")
print("   those 26 specs failed before checkpoint write — re-run --resume")
print("   will retry them.")
PY
