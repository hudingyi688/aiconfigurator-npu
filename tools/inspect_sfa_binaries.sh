#!/usr/bin/env bash
# Inspect ALL prebuilt npu_sparse_flash_attention kernel variants in CANN's
# OPP dir. For each .json, print:
#   - inputs (name, dtype, paramType)
#   - attrs list
#   - outputs
#   - simplifiedKey (distinguishes variants)
#   - staticKey (hash over full signature)
# So we can see which dtype / attr combination actually has a prebuilt binary,
# and why our current bf16 + 5-attr call dispatches to none of them.
#
# Usage (on the NPU host):
#     tools/inspect_sfa_binaries.sh
#
# Env override:
#     SFA_DIR=/other/path tools/inspect_sfa_binaries.sh
#     OP_NAME=sparse_flash_attention   # or kv_quant_sparse_flash_attention

set -u

CANN_DIR="${CANN_DIR:-/usr/local/Ascend/cann-8.5.0}"
OP_NAME="${OP_NAME:-sparse_flash_attention}"
SFA_DIR="${SFA_DIR:-${CANN_DIR}/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_transformer/${OP_NAME}}"

echo "=== scanning: $SFA_DIR ==="
if [ ! -d "$SFA_DIR" ]; then
    echo "ERROR: directory not found: $SFA_DIR" >&2
    exit 1
fi

count=0
for f in "$SFA_DIR"/*.json; do
    [ -f "$f" ] || continue
    count=$((count + 1))
    echo
    echo "=============================================================="
    echo "Variant $count : $(basename "$f")"
    echo "=============================================================="
    python3 - "$f" <<'PYEOF'
import json, sys
f = sys.argv[1]
try:
    d = json.load(open(f))
except Exception as e:
    print(f"  (parse error: {e})")
    sys.exit(0)

s = d.get("supportInfo", {})

print("kernelName   :", d.get("kernelName", "?"))
print("opMode       :", s.get("opMode", "?"))
print("implMode     :", s.get("implMode", "?"))
print("int64Mode    :", s.get("int64Mode", "?"))

print("inputs       :")
for i in s.get("inputs", []):
    print(f"    [{i.get('index')}] {i.get('name'):<28} dtype={i.get('dtype'):<10} "
          f"paramType={i.get('paramType','')} "
          f"format={i.get('format')}")

print("outputs      :")
for o in s.get("outputs", []):
    print(f"    [{o.get('index')}] {o.get('name'):<28} dtype={o.get('dtype'):<10}")

print("attrs        :")
for a in s.get("attrs", []):
    print(f"    {a.get('name'):<28} dtype={a.get('dtype'):<10} "
          f"default={a.get('value')!r}")

keys = s.get("simplifiedKey", [])
print(f"simplifiedKey ({len(keys)} entries):")
for k in keys:
    print(f"    {k}")

print("staticKey    :", s.get("staticKey", "?"))

kernel_list = d.get("kernelList", [])
print(f"kernelList   : {len(kernel_list)} tiling variants")
tilings = sorted({kl.get("tilingKey") for kl in kernel_list})
print(f"    tilingKeys = {tilings}")
PYEOF
done

echo
echo "=============================================================="
echo "Total variants: $count"
echo
echo "Also dumping the top-level dispatch config (tells the runtime which"
echo "binary to match to a given call):"
echo "=============================================================="
CFG="${CANN_DIR}/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_transformer/${OP_NAME}.json"
if [ -f "$CFG" ]; then
    python3 - "$CFG" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
lst = cfg.get("binList", [])
print(f"binList entries: {len(lst)}")
for i, b in enumerate(lst):
    print(f"--- bin {i} ---")
    ins = b.get("inputs", [])
    attrs = b.get("attrs", [])
    print("  input dtypes :", [x.get("dtype") for x in ins])
    print("  attrs        :", [a.get("name") for a in attrs])
    keys = b.get("simplifiedKey", [])
    print(f"  simplifiedKey: {keys}")
PYEOF
else
    echo "  (dispatch config not found at $CFG)"
fi
