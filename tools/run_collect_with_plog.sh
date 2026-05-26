#!/usr/bin/env bash
# Run collect_mla_module --quick and grep CANN plog around the run for
# the real op-level error code. ASCEND_LAUNCH_BLOCKING=1 is rejected by
# vllm-ascend (ACL graph incompatibility), so we use the plog approach
# that vllm-ascend itself recommends.
#
# Usage on the NPU host:
#     bash tools/run_collect_with_plog.sh
#     bash tools/run_collect_with_plog.sh --force-mla
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

EXTRA="${*:-}"
LABEL="sfa"
[[ "${EXTRA}" == *"--force-mla"* ]] && LABEL="mla"

PLOG_DIR="${HOME}/ascend/log/debug/plog"
RUN_LOG="/tmp/collect_${LABEL}_run.log"
PLOG_OUT="/tmp/collect_${LABEL}_plog.txt"

mkdir -p "${PLOG_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
unset ASCEND_LAUNCH_BLOCKING || true

echo "[1/4] Snapshot existing plog files (for diff)..."
ls -1t "${PLOG_DIR}" 2>/dev/null > /tmp/plog_before.txt || true

echo "[2/4] Running collect_mla_module --quick ${EXTRA}"
timeout 600 \
    python3 -X faulthandler -u collector/npu/collect_mla_module.py \
        --mode context --quick \
        --batch-size 1 --seq-len 512 \
        --output-dir ./data/glm5_dsa_module \
        ${EXTRA} \
        > "${RUN_LOG}" 2>&1 \
    || true
rc=$?
echo "  run rc=${rc} ($(wc -l < "${RUN_LOG}") lines)  log: ${RUN_LOG}"

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
    tail -500 "${fp}" >> "${PLOG_OUT}"
done
echo "  plog excerpt: ${PLOG_OUT}  ($(wc -l < "${PLOG_OUT}") lines)"

echo "[4/4] Suspected sub-error lines (op name / 56xxxx / EZ / EE):"
echo "------- filtered plog tail -------"
grep -E "ERR|FAIL|ACL_ERROR|EZ[0-9]+|EE[0-9]+|56[0-9]{4}|kernel_name|fault kernel|func_name|OpAttr|InferShape|GetTilingFunc|aclnnSparse|aclnnFusedInfer|sparse_flash_attention|RingMLA|AtbRingMLA|head_dim|HeadSize|HeadNum|ShapeCheck|DimCheck|OutOfRange|NPU_ERROR|Address" "${PLOG_OUT}" \
    | head -80 \
    || echo "(no matching lines)"

echo
echo "------- last 30 lines of run log -------"
tail -30 "${RUN_LOG}"

echo
echo "Done. Useful files:"
echo "  ${RUN_LOG}"
echo "  ${PLOG_OUT}"
