#!/usr/bin/env bash
# Collect CANN plog around an SFA failure to extract the real
# sub-error code that ACL 561002 / RingMLA / RingSFA wraps.
#
# Strategy:
#   1. Clear / rotate any old plog files so we only capture this run.
#   2. Run the standalone SFA reproducer (already in tools/) so the
#      kernel fails in a controlled way.
#   3. Tail the most recent plog files looking for:
#         [ERROR] OPP / AICORE / RUNTIME / TBE
#         "kernel_name", "InferShape", "OpAttr", "GetTilingFunc"
#         "fault kernel", "func_name", "ACL_ERROR_*"
#   4. Print a summary so we can decide whether SFA's failure is the
#      same OPP-prebuilt-missing wall as FIA(head_dim=192), or
#      something fixable in code (KV layout, slot_mapping, topk).
#
# Usage on the NPU host:
#     tools/collect_plog_for_sfa.sh
set -u

cd "$(dirname "$0")/.." || exit 1

PLOG_DIR="${HOME}/ascend/log/debug/plog"
RUN_LOG="/tmp/sfa_run.log"
PLOG_OUT="/tmp/sfa_plog.txt"

mkdir -p "${PLOG_DIR}"

echo "[1/4] Snapshot existing plog file list (will diff after run)..."
ls -1t "${PLOG_DIR}" 2>/dev/null > /tmp/plog_before.txt || true

echo "[2/4] Running tools/probe_sfa_standalone.sh ..."
bash tools/probe_sfa_standalone.sh > "${RUN_LOG}" 2>&1 || true
echo "    run log:  ${RUN_LOG}  ($(wc -l < "${RUN_LOG}") lines)"

echo "[3/4] Diffing plog files to find the new one(s)..."
ls -1t "${PLOG_DIR}" 2>/dev/null > /tmp/plog_after.txt || true
NEW_FILES=$(comm -23 <(sort /tmp/plog_after.txt) <(sort /tmp/plog_before.txt))

if [[ -z "${NEW_FILES}" ]]; then
  echo "    No new plog files — falling back to most recent file."
  NEW_FILES=$(ls -1t "${PLOG_DIR}" 2>/dev/null | head -1)
fi

: > "${PLOG_OUT}"
for f in ${NEW_FILES}; do
  fp="${PLOG_DIR}/${f}"
  [[ -f "${fp}" ]] || continue
  echo "===== ${fp} =====" >> "${PLOG_OUT}"
  tail -300 "${fp}" >> "${PLOG_OUT}"
done

echo "    plog excerpt: ${PLOG_OUT}  ($(wc -l < "${PLOG_OUT}") lines)"

echo "[4/4] Extracting suspected sub-error lines..."
echo "------- 561 / EZ / EE codes & op names -------"
grep -E "ERR|FAIL|ACL_ERROR|EZ[0-9]+|EE[0-9]+|56[0-9]{4}|kernel_name|fault kernel|func_name|OpAttr|InferShape|GetTilingFunc|aclnnSparse|aclnnFusedInfer|sparse_flash_attention|RingMLA" "${PLOG_OUT}" \
  | head -120 \
  || echo "(no matching lines)"

echo ""
echo "Done."
echo "  Run output  : ${RUN_LOG}"
echo "  Plog excerpt: ${PLOG_OUT}"
echo ""
echo "Tail of run log:"
tail -40 "${RUN_LOG}"
