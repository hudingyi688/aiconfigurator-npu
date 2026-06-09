#!/usr/bin/env bash
# Collect HCCL broadcast silicon data for KV transfer modeling.
#
# WHY: KV transfer (87% of GLM-5 disagg prefill) is modeled from a 6-point
# profiler-derived table (ep∈{1,16} × isl∈{2.5k,10k,20k}). isl>20k currently
# uses linear extrapolation. Replacing the extrapolation with a physics-based
# model requires broadcast latency data across message_bytes and num_devices.
# The existing hcom_broadcast_.csv has only 1 point (1MB, 32dev), insufficient
# for interpolation.
#
# WHAT IS COLLECTED: hcom_broadcast_ kernel Duration (us) via CANN profiler
# → kernel_details.csv → aggregated into hcom_broadcast_.csv.
# Message sizes: 128B~512MB powers-of-2 (23 points).
# Device groups: nd=2/4/8/16 on single node, nd=32 with 2 nodes.
#
# MULTI-NODE (nd=32, inter-pod): run this SAME script on BOTH nodes
# simultaneously with distinct NODE_RANK values:
#   NODE_RANK=0 on master node (also set MASTER_ADDR to its IP)
#   NODE_RANK=1 on second node
#
# USAGE:
#   # Single-node (nd=16/8/4/2):
#   bash collect_broadcast.sh [OUTPUT_DIR]
#
#   # Multi-node nd=32 (run simultaneously on both nodes):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash collect_broadcast.sh ./broadcast_inter_pod
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash collect_broadcast.sh ./broadcast_inter_pod
#
# OUTPUT: <OUTPUT_DIR>/hcom_broadcast_.csv
# Merge this file into:
#   systems/data/ascend_910b/vllm-ascend/0.18.0/hcom_broadcast_.csv
# (create the hccl/v8.5/ sub-directory if it does not yet exist in the DB path)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/generate_comm_microbench.py"

OUTPUT_DIR="${1:-./hccl_broadcast_data}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29700}"
NPROC="${NPROC:-}"   # devices per node; auto-detected if empty

mkdir -p "${OUTPUT_DIR}"

# Auto-detect devices per node (A3 = 16 DIE = 16 NPU devices).
if [[ -z "${NPROC}" ]]; then
    NPROC="$(python3 -c 'import torch, torch_npu; print(torch.npu.device_count())' 2>/dev/null || echo "")"
    if [[ -z "${NPROC}" || "${NPROC}" == "0" ]]; then
        echo "[ERROR] could not auto-detect NPU device count; set NPROC explicitly (A3=16)." >&2
        exit 1
    fi
    echo "[collect-broadcast] auto-detected NPROC=${NPROC} devices/node"
fi

WORLD=$(( NNODES * NPROC ))

echo "======================================="
echo "[collect-broadcast] HCCL broadcast collection"
echo "  NNODES    : ${NNODES}  (world_size=${WORLD})"
echo "  NODE_RANK : ${NODE_RANK}"
echo "  MASTER    : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  NPROC     : ${NPROC}"
echo "  OUTPUT    : ${OUTPUT_DIR}"
echo "  SCRIPT    : ${BENCH_SCRIPT}"
echo "======================================="

# Message bytes grid: 128B ~ 512MB powers-of-2 (23 points, same as run_comm_bench.sh)
MSG_BYTES="128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 \
1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912"

run_broadcast() {
    local ndev="$1"
    local port="$2"
    local tier="$3"

    if (( WORLD < ndev )); then
        echo "[collect-broadcast] skip nd=${ndev}: world_size=${WORLD} < nd"
        return 0
    fi

    echo ""
    echo "--- broadcast nd=${ndev}  tier=${tier}  port=${port}  $(date '+%H:%M:%S') ---"

    set +e
    torchrun \
        --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
        --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" --master_port="${port}" \
        "${BENCH_SCRIPT}" \
        --do-run \
        --bench-mode kernel \
        --ops broadcast \
        --grid-shape 48 8 2 \
        --num-devices "${ndev}" \
        --topology-tier "${tier}" \
        --bytes-grid ${MSG_BYTES} \
        --output-dir "${OUTPUT_DIR}"
    local rc=$?
    set -e

    if [[ ${rc} -eq 0 ]]; then
        echo "    OK ($(date '+%H:%M:%S'))"
    elif [[ ${rc} -eq 139 ]]; then
        echo "    SIGSEGV at shutdown (known torch_npu issue, data is safe)"
    else
        echo "    WARNING: exit code ${rc}, continuing..." >&2
    fi
}

PORT="${MASTER_PORT}"

# tier=2 (die-level, intra-node 2 devices)
run_broadcast 2  $(( PORT + 1 )) 2
# tier=1 (intra-pod, intra-node ≥4 devices)
run_broadcast 4  $(( PORT + 2 )) 1
run_broadcast 8  $(( PORT + 3 )) 1
run_broadcast 16 $(( PORT + 4 )) 1
# tier=0 (inter-pod, needs 2 nodes / world_size=32)
run_broadcast 32 $(( PORT + 5 )) 0

echo ""
echo "======================================="
echo "[collect-broadcast] Collection complete"
echo "  End time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output   : ${OUTPUT_DIR}"
echo "======================================="
ls -lh "${OUTPUT_DIR}/" 2>/dev/null || true
echo ""
wc -l "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Next step: merge hcom_broadcast_.csv into the HCCL data directory:"
echo "  systems/data/ascend_910b/vllm-ascend/0.18.0/hcom_broadcast_.csv"
