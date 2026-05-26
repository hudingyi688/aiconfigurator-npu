#!/usr/bin/env bash
# Confirm whether vllm-ascend's OOT custom op pack actually contains its
# own SparseFlashAttention bin, and whether the OOT bin's attr list
# matches sfa_v1.py's call signature (5 attrs: scale_value,
# sparse_block_size, layout_query, layout_kv, sparse_mode).
#
# Hypothesis: ACL is dispatching the 5-attr vllm-ascend call to the
# 9-attr CANN-shipped kernel, causing "attr index 5/6/7/8 out of range 5".
#
# Usage on the NPU host:
#     bash tools/inspect_oot_sfa_bins.sh
set -u

OOT="/usr/local/python3.11.14/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
echo "=== OOT vendor root ==="
echo "  ${OOT}"
ls -la "${OOT}" 2>/dev/null || { echo "ERROR: vendor dir not found"; exit 1; }

echo
echo "=== Layout under op_impl / op_proto / op_api ==="
for sub in op_impl op_proto op_api; do
    echo "----- ${sub} -----"
    find "${OOT}/${sub}" -maxdepth 6 -type d 2>/dev/null | head -20
done

echo
echo "=== Search for SparseFlashAttention json descriptors in OOT ==="
find "${OOT}" -name "*.json" -path "*[Ss]parse[Ff]lash*" 2>/dev/null \
    -o -name "SparseFlashAttention*" 2>/dev/null \
    | head -20

echo
echo "=== Any *.o / *.json under OOT (head) ==="
find "${OOT}" \( -name "*.json" -o -name "*.o" \) 2>/dev/null | head -30

echo
echo "=== Inspect any SparseFlashAttention .json under OOT ==="
for f in $(find "${OOT}" -name "*[Ss]parse[Ff]lash*.json" 2>/dev/null); do
    echo "----- $f -----"
    python3 - "$f" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  parse error: {e}")
    sys.exit(0)
s = d.get("supportInfo", {})
print(f"  kernelName : {d.get('kernelName')}")
print(f"  inputs ({len(s.get('inputs', []))}) :")
for i in s.get("inputs", []):
    print(f"    [{i.get('index')}] {i.get('name'):<28} dtype={i.get('dtype')}")
print(f"  attrs  ({len(s.get('attrs', []))}) :")
for a in s.get("attrs", []):
    print(f"    {a.get('name'):<28} dtype={a.get('dtype')}  default={a.get('value')!r}")
keys = s.get("simplifiedKey", [])
print(f"  simplifiedKey ({len(keys)}):")
for k in keys[:6]:
    print(f"    {k}")
PYEOF
done

echo
echo "=== Inspect OOT dispatch config (top-level config json) ==="
for cfg in $(find "${OOT}" -path "*config*sparse_flash*.json" -o -path "*sparse_flash*config*.json" 2>/dev/null | head -5); do
    echo "----- $cfg -----"
    python3 - "$cfg" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
binList = cfg.get("binList", [])
print(f"  binList entries: {len(binList)}")
for i, b in enumerate(binList[:6]):
    print(f"  --- bin {i} ---")
    print(f"    input dtypes : {[x.get('dtype') for x in b.get('inputs', [])]}")
    print(f"    attrs        : {[a.get('name') for a in b.get('attrs', [])]}")
PYEOF
done

echo
echo "=== Compare vs CANN OPP shipped SFA bin (for reference) ==="
CANN_SFA="/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_transformer/sparse_flash_attention"
echo "  ${CANN_SFA}"
ls "${CANN_SFA}"/*.json 2>/dev/null | head -5
echo
echo "  CANN-shipped attrs (sample first .json):"
for f in $(ls "${CANN_SFA}"/*.json 2>/dev/null | head -1); do
    python3 - "$f" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("supportInfo", {})
print(f"    attrs ({len(s.get('attrs', []))}):")
for a in s.get("attrs", []):
    print(f"      {a.get('name')}")
PYEOF
done

echo
echo "=== ASCEND_CUSTOM_OPP_PATH effect on dispatch order ==="
echo "  current : ${ASCEND_CUSTOM_OPP_PATH:-(unset)}"
echo "  if OOT comes first, OOT bin should win — but we're seeing the"
echo "  CANN-shipped 9-attr signature, suggesting OOT either lacks the"
echo "  bin entirely OR gets overridden by CANN dispatch."
