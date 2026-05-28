#!/usr/bin/env bash
# Sweep MoE dispatch+combine across ep_size ∈ {2,4,8,16,32}.
#
# Each ep_size is one torchrun invocation. We sweep ep serially (not
# concurrently) because each invocation owns the NPU device set.
#
# The outermost torchrun reads MASTER_ADDR / MASTER_PORT / NNODES /
# NODE_RANK / NPROC_PER_NODE from env vars — set them according to
# your job scheduler (slurm srun / k8s job / manual ssh-launch).
#
# For 2 nodes × 16 NPUs = 32 ranks, use --nnodes=2 --node_rank={0,1}
# with the same MASTER_ADDR.
#
# Usage on the NPU host (32-rank, 2 nodes, current node is rank 0):
#     export MASTER_ADDR=<head_node_ip>
#     export MASTER_PORT=29500
#     export NNODES=2
#     export NODE_RANK=0          # 1 on the second node
#     export NPROC_PER_NODE=16
#     bash tools/run_moe_dispatch_sweep.sh
#
# Single-node 8-card test:
#     export MASTER_ADDR=127.0.0.1
#     export MASTER_PORT=29500
#     export NNODES=1
#     export NODE_RANK=0
#     export NPROC_PER_NODE=8
#     bash tools/run_moe_dispatch_sweep.sh
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_RANKS=$(( NNODES * NPROC_PER_NODE ))

# GLM-5 shape (override via env)
HIDDEN="${HIDDEN:-6144}"
INTER="${INTER:-2048}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
TOPK="${TOPK:-8}"

# Token grid (production-derived; tweak as needed)
NUM_TOKENS_LIST="${NUM_TOKENS_LIST:-1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096}"

# Quant modes
QUANT_TYPES="${QUANT_TYPES:-bf16 w8a8_dynamic}"

# EP sweep — only sizes that divide the total rank count
ALL_EP_SIZES="${EP_SIZES:-2 4 8 16 32}"
EP_SIZES_VALID=""
for ep in ${ALL_EP_SIZES}; do
    if (( TOTAL_RANKS % ep == 0 )) && (( ep <= TOTAL_RANKS )); then
        EP_SIZES_VALID="${EP_SIZES_VALID} ${ep}"
    fi
done
EP_SIZES_VALID=$(echo "${EP_SIZES_VALID}" | xargs)
echo "[sweep] total_ranks=${TOTAL_RANKS}  ep_sizes=${EP_SIZES_VALID}"

OUT_DIR="${OUT_DIR:-./data/moe_dispatch_combine}"
mkdir -p "${OUT_DIR}"

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

count=0
for ep in ${EP_SIZES_VALID}; do
    count=$((count + 1))
    echo
    echo "================================================================"
    echo "[${count}] sweep ep_size=${ep}  $(date +%T)"
    echo "================================================================"

    LOG="${OUT_DIR}/run_ep${ep}_node${NODE_RANK}.log"
    : > "${LOG}"

    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        collector/npu/collect_moe_dispatch_combine.py \
            --hidden "${HIDDEN}" \
            --inter "${INTER}" \
            --num-experts "${NUM_EXPERTS}" \
            --topk "${TOPK}" \
            --ep-size "${ep}" \
            --num-tokens-list ${NUM_TOKENS_LIST} \
            --quant-types ${QUANT_TYPES} \
            --output-dir "${OUT_DIR}" \
        2>&1 | tee "${LOG}"
    rc=${PIPESTATUS[0]}
    echo "[ep=${ep}] rc=${rc} log=${LOG}"
done

echo
echo "================================================================"
echo "Done. Output: ${OUT_DIR}/"
echo "================================================================"
ls -la "${OUT_DIR}/"
