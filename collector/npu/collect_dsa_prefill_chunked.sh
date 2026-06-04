#!/usr/bin/env bash
# Collect GLM-5 DSA PREFILL silicon for the production (chunked + CP) shape.
#
# WHY these flags (derived from production config, see memory aiconfigurator-npu):
#   - GLM-5 DSA prefill uses Context Parallelism (sequence-dim split), NOT
#     tensor-parallel head split. Every rank keeps ALL 64 heads.
#         => --num-heads-override 64   (NOT 64//tp; verified: tp16 profiler nh=64)
#   - Production chunked prefill: max_num_batched_tokens=4096, CP16
#     => each rank processes 4096/16 = 256 NEW query tokens per SFA step,
#        attending to the cumulative KV (seq_len).
#         => --chunk-query-len 256     (verified: profiler SFA shape first dim=256)
#   - Production weights are W8A8 (model dir GLM-5-w8a8, serve --quantization ascend)
#         => --quantization ascend     (the old float16 table is wrong)
#
# The seq_len sweep (cumulative KV points) is the script default and now
# includes 20480 so the 20k prefill's last chunk is covered.
#
# PYTHONPATH fix: GE's C++-layer TBE import needs CANN site-packages on
# PYTHONPATH (tbe is importable at top level via .pth, but the GE subprocess
# does not inherit .pth). Without this you get "AclSetCompileopt 500001 /
# Failed to init tbe". Adjust the CANN path to your install if different.
set -euo pipefail

# --- adjust these to your NPU box --------------------------------------------
MODEL_DIR="${MODEL_DIR:-/mnt/deepseek/lwy/model/GLM-5-w8a8}"
CANN_SITE="${CANN_SITE:-/usr/local/Ascend/cann-8.5.0/python/site-packages}"
OUT_DIR="${OUT_DIR:-./dsa_prefill_chunked_nh64_w8a8}"
DEVICE="${DEVICE:-npu:0}"
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${CANN_SITE}:${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[collect-dsa-prefill] model=${MODEL_DIR}"
echo "[collect-dsa-prefill] PYTHONPATH=${PYTHONPATH}"
echo "[collect-dsa-prefill] out=${OUT_DIR}"

python3 "${SCRIPT_DIR}/collect_mla_module.py" \
  --mode context \
  --model "${MODEL_DIR}" \
  --num-heads-override 64 \
  --chunk-query-len 256 \
  --quantization ascend \
  --output-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --resume

echo "[collect-dsa-prefill] DONE. Verify: rows should have num_heads=64,"
echo "  gemm_type=w8a8_dynamic, and latency should SATURATE for isl>=8192"
echo "  (DSA sparse topk~2048 caps it). Then merge into"
echo "  systems/data/ascend_910b/vllm-ascend/0.18.0/dsa_context_module_perf.txt"
