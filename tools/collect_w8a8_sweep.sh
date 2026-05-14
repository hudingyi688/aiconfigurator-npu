#!/usr/bin/env bash
# Add W8A8 dynamic sweep on top of an existing BF16 collection without
# losing or re-running anything.
#
# Strategy:
#   1. Tar-backup data/{gemm,moe,elementwise}/ to data/backups/<ts>.
#   2. Run gemm + moe with `--quant-types bf16 w8a8_dynamic --resume`
#      into the same dirs. The existing checkpoint files are keyed
#      on (m,n,k,quant_type) / (...,quant_type,model), so BF16 specs
#      are skipped and only new W8A8 specs run.
#   3. Skip elementwise — it has no quant axis and was already done.
#
# Usage on the NPU host:
#     tools/collect_w8a8_sweep.sh
#     OUT_ROOT=data tools/collect_w8a8_sweep.sh   # explicit
#
# To pass extra args to both gemm + moe, append after `--`:
#     tools/collect_w8a8_sweep.sh -- --m-list 1 16 128
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
unset VLLM_ASCEND_ENABLE_MLAPO
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

OUT_ROOT="${OUT_ROOT:-data}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${OUT_ROOT}/backups"
BACKUP_TAR="${BACKUP_DIR}/pre_w8a8_${TS}.tar.gz"
LOG="${OUT_ROOT}/run_w8a8.log"

EXTRA=""
while (( "$#" )); do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA="$*"
    break
  fi
  shift
done

mkdir -p "${BACKUP_DIR}"

echo "[1/3] Backing up existing data to ${BACKUP_TAR} ..."
if compgen -G "${OUT_ROOT}/gemm/*.csv" > /dev/null \
   || compgen -G "${OUT_ROOT}/moe/*.csv" > /dev/null \
   || compgen -G "${OUT_ROOT}/elementwise/*.csv" > /dev/null; then
  tar --exclude="${OUT_ROOT}/backups" -czf "${BACKUP_TAR}" \
      "${OUT_ROOT}/gemm" "${OUT_ROOT}/moe" "${OUT_ROOT}/elementwise" \
      2>/dev/null \
    && echo "    backup OK ($(du -h "${BACKUP_TAR}" | cut -f1))" \
    || { echo "    backup FAILED — aborting"; exit 1; }
else
  echo "    nothing to back up (no existing csvs) — proceeding"
fi

: > "${LOG}"

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

GEMM_RC=0; MOE_RC=0
echo "[2/3] gemm + moe with bf16 + w8a8_dynamic --resume ..."
run_one "GEMM" "collector/npu/collect_gemm.py" \
  --quant-types bf16 w8a8_dynamic \
  --output-dir "${OUT_ROOT}/gemm" --resume || GEMM_RC=$?
run_one "MOE" "collector/npu/collect_moe.py" \
  --quant-types bf16 w8a8_dynamic \
  --output-dir "${OUT_ROOT}/moe" --resume || MOE_RC=$?

echo "==============================================================" | tee -a "${LOG}"
echo "[3/3] Summary:" | tee -a "${LOG}"
echo "  backup       : ${BACKUP_TAR}" | tee -a "${LOG}"
echo "  gemm         : rc=${GEMM_RC}    output=${OUT_ROOT}/gemm" | tee -a "${LOG}"
echo "  moe          : rc=${MOE_RC}    output=${OUT_ROOT}/moe" | tee -a "${LOG}"
echo "  elementwise  : skipped (no quant axis, already done)" | tee -a "${LOG}"
echo "  combined log : ${LOG}" | tee -a "${LOG}"
echo "==============================================================" | tee -a "${LOG}"

if (( GEMM_RC == 0 || MOE_RC == 0 )); then
  exit 0
else
  exit "${GEMM_RC}"
fi
