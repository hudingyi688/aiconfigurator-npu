#!/usr/bin/env bash
# Collect the FusedMC2 dispatch+FFN+combine silicon at a target EP size.
#
# WHY: the perf DB only has ep2/4/8 (moe_dispatch_combine_perf.txt); production
# GLM-5 runs ep32, currently approximated by clamping to ep8. This fills larger
# ep so queries hit measured data instead of clamping.
#
# CONSTRAINT (moe_dispatch_factory.setup_distributed): the EP group is a REAL
# HCCL subgroup of the torchrun world, so WORLD_SIZE % ep_size == 0 AND every
# rank is a PHYSICAL device. You cannot emulate a larger ep on fewer devices.
#
# A3 (Atlas 800 A3): one node exposes 16 DIE as 16 NPU devices
# (torch.npu.device_count()==16). So device math is per-DIE:
#   ep16  -> WORLD=16 -> ONE A3 node           (doable NOW, no extra nodes)
#   ep32  -> WORLD=32 -> TWO A3 nodes (2x16)    (needs 1 extra node, not 4)
# If device_count()==8 on your box (per-card, not per-DIE), set GPUS=8:
#   ep8 single node, ep32 -> 4 nodes.
#
# Multi-node: run this SAME script on every node with the same
# MASTER_ADDR/MASTER_PORT and a distinct NODE_RANK (0..NNODES-1).
#
# RECOMMENDED while waiting for a 2nd node: EP=16 NNODES=1 (single A3 node).
# ep16 is much closer to production ep32 than the current ep8 clamp.
set -euo pipefail

# --- adjust per run / per node ----------------------------------------------
EP="${EP:-16}"                       # target ep_size (16 = single A3 node; 32 = 2 A3 nodes)
GPUS="${GPUS:-}"                     # NPU devices per node; auto-detect if empty (A3=16)
NNODES="${NNODES:-1}"                # nodes (ep must divide NNODES*GPUS)
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

# Auto-detect NPU devices per node if GPUS not set (A3 = 16 DIE = 16 devices).
if [[ -z "${GPUS}" ]]; then
  GPUS="$(python3 -c 'import torch, torch_npu; print(torch.npu.device_count())' 2>/dev/null || echo "")"
  if [[ -z "${GPUS}" || "${GPUS}" == "0" ]]; then
    echo "[ERROR] could not auto-detect NPU device count; set GPUS explicitly (A3=16)." >&2
    exit 1
  fi
  echo "[collect-moe-dispatch] auto-detected GPUS=${GPUS} devices/node"
fi

WORLD=$(( NNODES * GPUS ))
if (( WORLD % EP != 0 )); then
  echo "[ERROR] WORLD_SIZE=${WORLD} (NNODES=${NNODES} x GPUS=${GPUS}) must be a multiple of EP=${EP}." >&2
  echo "        A3 has 16 DIE/node: ep16 -> NNODES=1, ep32 -> NNODES=2." >&2
  exit 1
fi
if (( WORLD < EP )); then
  echo "[ERROR] WORLD_SIZE=${WORLD} < EP=${EP}: EP ranks must be physical devices, add nodes." >&2
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
