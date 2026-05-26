#!/usr/bin/env bash
# Run collect_mla_module --quick with ASCEND_LAUNCH_BLOCKING=1 so any
# kernel failure surfaces with an accurate Python stacktrace instead
# of being delayed to the next synchronize.
#
# Two passes:
#   pass A: SFA path (default, --quick)
#   pass B: force-mla path (--force-mla)
#
# Usage on the NPU host:
#     bash tools/run_collect_blocking.sh
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export ASCEND_LAUNCH_BLOCKING=1

run () {
    local label="$1"; shift
    local log="/tmp/collect_${label}.log"
    : > "$log"
    echo "=============================================================="
    echo "[$label] starting (logs -> $log)"
    echo "=============================================================="
    timeout 600 \
        python3 -X faulthandler -u collector/npu/collect_mla_module.py \
            --mode context --quick \
            --batch-size 1 --seq-len 512 \
            --output-dir ./data/glm5_dsa_module \
            "$@" \
            2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo
    echo "[$label] EXIT_CODE: $rc"
    echo "=== last 60 lines ==="
    tail -60 "$log"
    echo
}

run "blocking_sfa"
run "blocking_mla" --force-mla

echo
echo "Logs:"
echo "  /tmp/collect_blocking_sfa.log"
echo "  /tmp/collect_blocking_mla.log"
