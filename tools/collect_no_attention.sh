#!/usr/bin/env bash
# Run the attention-free collectors (gemm + moe + elementwise) in
# sequence on this NPU box. Skips MLA / SFA on purpose: ACL plog
# confirmed errno[561003] OpName:[SparseFlashAttention_*] "The binary
# bin not found!" — this CANN-OPP install is missing the prebuilt
# binaries SFA and FIA(head_dim>128) need. The remaining collectors
# don't depend on those binaries.
#
# Usage on the NPU host:
#     tools/collect_no_attention.sh                     # default args
#     tools/collect_no_attention.sh --quant-types bf16 w8a8_dynamic
#     tools/collect_no_attention.sh --resume            # gemm/moe both
#                                                      # support --resume
#
# Pass-through: any extra args after `--` go to *each* collector.
# Per-collector overrides:
#     GEMM_ARGS / MOE_ARGS / EW_ARGS environment variables.
#
# Output:
#     data/gemm/         (gemm csv + checkpoint)
#     data/moe/
#     data/elementwise/
#     data/run.log       (combined stdout/stderr, tail -f-friendly)
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
unset VLLM_ASCEND_ENABLE_MLAPO
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-data}"
mkdir -p "${OUT_ROOT}/gemm" "${OUT_ROOT}/moe" "${OUT_ROOT}/elementwise"
LOG="${OUT_ROOT}/run.log"
: > "${LOG}"

# Default per-collector arg sets — kept conservative; override via env.
GEMM_ARGS_DEFAULT="--quant-types bf16 --output-dir ${OUT_ROOT}/gemm"
MOE_ARGS_DEFAULT="--quant-types bf16 --output-dir ${OUT_ROOT}/moe"
EW_ARGS_DEFAULT="--output-dir ${OUT_ROOT}/elementwise"

GEMM_ARGS="${GEMM_ARGS:-${GEMM_ARGS_DEFAULT}}"
MOE_ARGS="${MOE_ARGS:-${MOE_ARGS_DEFAULT}}"
EW_ARGS="${EW_ARGS:-${EW_ARGS_DEFAULT}}"

# Anything after `--` is appended to every collector (e.g. --resume).
EXTRA=""
while (( "$#" )); do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA="$*"
    break
  fi
  shift
done

run_one() {
  local label="$1"; shift
  local script="$1"; shift
  local args="$*"
  echo "==============================================================" | tee -a "${LOG}"
  echo "[${label}] python ${script} ${args} ${EXTRA}" | tee -a "${LOG}"
  echo "==============================================================" | tee -a "${LOG}"
  local t0=$(date +%s)
  if python -X faulthandler "${script}" ${args} ${EXTRA} >> "${LOG}" 2>&1; then
    echo "[${label}] OK in $(( $(date +%s) - t0 ))s" | tee -a "${LOG}"
    return 0
  else
    local rc=$?
    echo "[${label}] FAILED rc=${rc} after $(( $(date +%s) - t0 ))s" | tee -a "${LOG}"
    return ${rc}
  fi
}

# We do not abort on a single collector failure — partial data is
# still useful, and gemm / moe / elementwise are independent.
GEMM_RC=0; MOE_RC=0; EW_RC=0
run_one "GEMM"        "collector/npu/collect_gemm.py"        ${GEMM_ARGS} || GEMM_RC=$?
run_one "MOE"         "collector/npu/collect_moe.py"         ${MOE_ARGS}  || MOE_RC=$?
run_one "ELEMENTWISE" "collector/npu/collect_elementwise.py" ${EW_ARGS}   || EW_RC=$?

echo "==============================================================" | tee -a "${LOG}"
echo "Summary:" | tee -a "${LOG}"
echo "  gemm        : rc=${GEMM_RC}    output=${OUT_ROOT}/gemm" | tee -a "${LOG}"
echo "  moe         : rc=${MOE_RC}    output=${OUT_ROOT}/moe" | tee -a "${LOG}"
echo "  elementwise : rc=${EW_RC}    output=${OUT_ROOT}/elementwise" | tee -a "${LOG}"
echo "  combined log: ${LOG}" | tee -a "${LOG}"
echo "==============================================================" | tee -a "${LOG}"

# Exit 0 if at least one succeeded, otherwise propagate the first failure.
if (( GEMM_RC == 0 || MOE_RC == 0 || EW_RC == 0 )); then
  exit 0
else
  exit "${GEMM_RC}"
fi
