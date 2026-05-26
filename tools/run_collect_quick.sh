#!/usr/bin/env bash
# Run collect_mla_module --quick with unbuffered output + faulthandler
# so we see exactly where it stops (segfault, hang, exception).
#
# Usage on the NPU host:
#     bash tools/run_collect_quick.sh
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

LOG="/tmp/collect_quick.log"
: > "$LOG"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Time-bound the run so a hang doesn't block forever (10 min)
timeout 600 \
    python3 -X faulthandler -u collector/npu/collect_mla_module.py \
        --mode context --quick \
        --batch-size 4 --seq-len 2048 \
        --output-dir ./data/glm5_dsa_module \
        2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo
echo "EXIT_CODE: $rc"
echo "  124 -> killed by timeout (hang)"
echo "  139 -> segfault"
echo "  0   -> success"
echo
echo "Log: $LOG"
echo
echo "=== last 5 lines ==="
tail -5 "$LOG"
