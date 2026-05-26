#!/usr/bin/env bash
# Test whether ASCEND_CUSTOM_OPP_PATH must be exported BEFORE python
# starts (i.e. before torch_npu / ACL runtime init reads it), as
# opposed to setting it from inside python via NPUPlatform.import_kernels().
#
# Hypothesis: real GLM-5 inference works because the user's shell has
# the path exported (or vllm-ascend's set_env.bash sourced) BEFORE
# python launches. Collector calls import_kernels() *after* torch_npu
# is imported — too late, ACL has already cached the dispatch search
# path.
#
# Usage on the NPU host:
#     bash tools/run_collect_with_oot_path.sh
#     bash tools/run_collect_with_oot_path.sh --force-mla
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# Source vllm-ascend's own set_env.bash if it exists (the install
# script that sets ASCEND_CUSTOM_OPP_PATH for OOT dispatch).
OOT_SET_ENV="/usr/local/python3.11.14/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash"
if [[ -f "${OOT_SET_ENV}" ]]; then
    echo "Sourcing ${OOT_SET_ENV}"
    # shellcheck disable=SC1090
    source "${OOT_SET_ENV}"
else
    echo "Manually exporting ASCEND_CUSTOM_OPP_PATH (set_env.bash not found)"
    export ASCEND_CUSTOM_OPP_PATH="/usr/local/python3.11.14/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH:-}"
fi
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"

EXTRA="${*:-}"
LABEL="sfa"
[[ "${EXTRA}" == *"--force-mla"* ]] && LABEL="mla"

PLOG_DIR="${HOME}/ascend/log/debug/plog"
RUN_LOG="/tmp/collect_oot_${LABEL}.log"
PLOG_OUT="/tmp/collect_oot_${LABEL}_plog.txt"

mkdir -p "${PLOG_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "[1/3] Snapshot existing plog files..."
ls -1t "${PLOG_DIR}" 2>/dev/null > /tmp/plog_before.txt || true

echo "[2/3] Running collect_mla_module --quick ${EXTRA}"
timeout 600 \
    python3 -X faulthandler -u collector/npu/collect_mla_module.py \
        --mode context --quick \
        --batch-size 1 --seq-len 512 \
        --output-dir ./data/glm5_dsa_module \
        ${EXTRA} \
        > "${RUN_LOG}" 2>&1 \
    || true
rc=$?
echo "  rc=${rc}  log lines: $(wc -l < ${RUN_LOG})"

echo "[3/3] Diff plog + filter sub-error lines"
ls -1t "${PLOG_DIR}" 2>/dev/null > /tmp/plog_after.txt || true
NEW_FILES=$(comm -23 <(sort /tmp/plog_after.txt) <(sort /tmp/plog_before.txt))
[[ -z "${NEW_FILES}" ]] && NEW_FILES=$(ls -1t "${PLOG_DIR}" 2>/dev/null | head -1)

: > "${PLOG_OUT}"
for f in ${NEW_FILES}; do
    fp="${PLOG_DIR}/${f}"
    [[ -f "${fp}" ]] || continue
    echo "===== ${fp} =====" >> "${PLOG_OUT}"
    tail -300 "${fp}" >> "${PLOG_OUT}"
done

echo "------- filtered plog tail -------"
grep -E "ERR|FAIL|ACL_ERROR|EZ[0-9]+|EE[0-9]+|56[0-9]{4}|kernel_name|fault kernel|func_name|OpAttr|InferShape|GetTilingFunc|aclnnSparse|aclnnFusedInfer|sparse_flash_attention|RingMLA|AtbRingMLA|attr|out of range|index|range" "${PLOG_OUT}" \
    | grep -v "TBE Subprocess" \
    | head -50 \
    || echo "(no matches)"

echo
echo "------- last 30 lines of run log -------"
tail -30 "${RUN_LOG}"

echo
echo "Files: ${RUN_LOG}  ${PLOG_OUT}"
