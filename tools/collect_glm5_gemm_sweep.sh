#!/usr/bin/env bash
# Sweep the rectangular GEMM (N, K) pairs that GLM-5 actually queries
# under aiconfigurator's DeepSeekV32Model decomposition.
#
# Existing gemm_perf.txt only has square (N=K) shapes from the
# DeepSeek-V3 baseline. None of GLM-5's real op shapes hit; HYBRID
# mode falls back to SOL+empirical for every GEMM call. This sweep
# fills the gap.
#
# Coverage (21 (N,K) pairs total, BF16 + W8A8):
#   dense MLP (layers 0..2, intermediate_size=12288):
#     gate_up TP={1,2,4,8}: (24576, 6144) (12288, 6144) (6144, 6144) (3072, 6144)
#     ffn2    TP={1,2,4,8}: (6144, 12288) (6144, 6144) (6144, 3072) (6144, 1536)
#   shared expert MLP (layers 3..77, moe_intermediate_size=2048):
#     gate_up TP={1,2,4,8}: (4096, 6144) (2048, 6144) (1024, 6144) (512, 6144)
#     ffn2    TP={1,2,4,8}: (6144, 2048) (6144, 1024) (6144, 512)  (6144, 256)
#   router:                 (256, 6144)
#   logits / lm_head TP={1,2,4,8}:
#                           (154880, 6144) (77440, 6144) (38720, 6144) (19360, 6144)
#
# M sweep matches the existing DEFAULT_M_LIST in collect_gemm.py
# (covers prefill chunks + decode batch sizes).
#
# Each (N, K) is a single collect_gemm invocation — that way the
# cartesian product reduces to exactly the pairs we want, and
# checkpoint resume means a crash mid-sweep is recoverable.
#
# Usage on the NPU host:
#     bash tools/collect_glm5_gemm_sweep.sh
#     bash tools/collect_glm5_gemm_sweep.sh /custom/data/root
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="${1:-data}"
OUT="${ROOT}/gemm_glm5"
LOG="${ROOT}/run_glm5_gemm.log"
mkdir -p "${OUT}"
: > "${LOG}"

NK_PAIRS=(
    # dense MLP gate_up (TP=1/2/4/8)
    "24576 6144"  "12288 6144"  "6144 6144"   "3072 6144"
    # dense MLP ffn2
    "6144 12288"  "6144 3072"   "6144 1536"
    # shared expert gate_up
    "4096 6144"   "2048 6144"   "1024 6144"   "512 6144"
    # shared expert ffn2
    "6144 2048"   "6144 1024"   "6144 512"    "6144 256"
    # router
    "256 6144"
    # logits / lm_head
    "154880 6144" "77440 6144"  "38720 6144"  "19360 6144"
)
# Notes:
#   - (6144, 6144) and (6144, 12288) overlap with existing square sweep
#     for some quant modes; --resume + spec_key dedup handles that.
#   - dense_ffn2 TP=2 is (6144, 6144) — covered above.

QUANT_TYPES=(bf16 w8a8_dynamic)

run_one () {
    local n="$1" k="$2"
    local label="N${n}_K${k}"
    local t0; t0=$(date +%s)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
        python3 collector/npu/collect_gemm.py \
            --output-dir "${OUT}" \
            --resume \
            --quant-types "${QUANT_TYPES[@]}" \
            --n-list "${n}" \
            --k-list "${k}" \
        2>&1 | tee -a "${LOG}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[${label}] FAILED rc=${rc} after $(($(date +%s) - t0))s" | tee -a "${LOG}"
    else
        echo "[${label}] OK in $(($(date +%s) - t0))s" | tee -a "${LOG}"
    fi
}

count=0
total=${#NK_PAIRS[@]}
for pair in "${NK_PAIRS[@]}"; do
    count=$((count + 1))
    set -- $pair
    n="$1"; k="$2"
    echo "============================================================"
    echo "[${count}/${total}] N=${n} K=${k}  $(date +%T)"
    echo "============================================================"
    run_one "${n}" "${k}"
done

echo
echo "============================================================"
echo "Done. Output: ${OUT}/"
echo "      Log   : ${LOG}"
echo "============================================================"
ls -la "${OUT}/"
echo
echo "Next step: append these rows into systems/data/.../gemm_perf.txt"
echo "  (the converter is tools/convert_to_lookup_table.py — run it"
echo "   pointing at this output dir, or hand-merge the new csv)."
