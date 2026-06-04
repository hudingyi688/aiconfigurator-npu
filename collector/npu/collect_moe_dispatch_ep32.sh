#!/usr/bin/env bash
# Collect the FusedMC2 dispatch+FFN+combine silicon at PRODUCTION EP32.
#
# WHY: the perf DB only has ep2/4/8 (moe_dispatch_combine_perf.txt); production
# GLM-5 runs ep32, currently approximated by clamping to ep8. This fills ep32.
#
# CONSTRAINT (moe_dispatch_factory.setup_distributed): WORLD_SIZE % ep_size == 0.
# So ep32 needs WORLD_SIZE >= 32, i.e. 4 nodes x 8 NPUs (NNODES=4, GPUS=8).
# This is a MULTI-NODE torchrun launch — run this SAME script on every node,
# with the same MASTER_ADDR/MASTER_PORT and a distinct NODE_RANK (0..NNODES-1).
#
# Single-node fallback: to (re)collect ep8 on one box, set EP=8 NNODES=1.
set -euo pipefail

# --- adjust per run / per node ----------------------------------------------
EP="${EP:-32}"                       # target ep_size (32 = production)
NNODES="${NNODES:-4}"                # nodes (ep32 needs NNODES*GPUS >= 32)
GPUS="${GPUS:-8}"                    # NPUs per node
NODE_RANK="${NODE_RANK:-0}"          # 0 on master, 1..NNODES-1 on others
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # master node IP (set on ALL nodes)
MASTER_PORT="${MASTER_PORT:-29512}"
OUT_DIR="${OUT_DIR:-./moe_dispatch_ep${EP}}"
TOKENS="${TOKENS:-1 2 4 8 16 32 64 128 256 512 1024 2048 4096}"
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # collector/ holds bench_engine
export PYTHONPATH="${COLLECTOR_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORLD=$(( NNODES * GPUS ))
if (( WORLD % EP != 0 )); then
  echo "[ERROR] WORLD_SIZE=${WORLD} (NNODES*GPUS) must be a multiple of EP=${EP}" >&2
  exit 1
fi

echo "[collect-moe-dispatch] EP=${EP} WORLD=${WORLD} (NNODES=${NNODES} x GPUS=${GPUS})"
echo "[collect-moe-dispatch] NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[collect-moe-dispatch] out=${OUT_DIR}  (rank-0 node writes moe_dispatch_combine_ep${EP}.csv)"

torchrun \
  --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/collect_moe_dispatch_combine.py" \
    --hidden 6144 --inter 2048 --num-experts 256 --topk 8 \
    --ep-size "${EP}" \
    --num-tokens-list ${TOKENS} \
    --quant-types bf16 w8a8_dynamic \
    --output-dir "${OUT_DIR}"

echo "[collect-moe-dispatch] DONE. Merge moe_dispatch_combine_ep${EP}.csv rows into"
echo "  systems/data/.../moe_dispatch_combine_perf.txt AND"
echo "  src/aiconfigurator_npu/systems/data/.../moe_dispatch_combine_perf.txt (both trees)."
echo "  Then ep32 queries hit it exactly instead of clamping to ep8."
