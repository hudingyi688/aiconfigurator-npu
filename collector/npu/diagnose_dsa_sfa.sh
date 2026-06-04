#!/usr/bin/env bash
# DIAGNOSE why DSA prefill silicon comes out ~100x too small.
#
# Hypothesis: the SparseFlashAttention kernel is NOT actually executing the
# sparse compute (binary-not-found 561003 -> silent degenerate path, OR the
# OOT custom-OPP path isn't injected so SFA no-ops). Symptom: latency ~1.5us
# at isl=4096 (profiler says ~282us/layer) and flat vs KV length.
#
# This runs ONE point (isl=8192, q=256, nh=64, w8a8) with AIC_PROBE_FIA=1 so
# the factory prints:
#   - attn_module type (should be AscendSFAImpl, NOT AscendMLAImpl)
#   - is_sparse (should be True)
#   - a standalone FIA/SFA self-test (passes => kernel present; fails with
#     561003 "binary bin not found" => SFA kernel missing on this CANN box)
#
# Read the [PROBE] lines in the output to settle it.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/deepseek/lwy/model/GLM-5-w8a8}"
CANN_SITE="${CANN_SITE:-/usr/local/Ascend/cann-8.5.0/python/site-packages}"
DEVICE="${DEVICE:-npu:0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${CANN_SITE}:${COLLECTOR_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export AIC_PROBE_FIA=1

echo "[diagnose] one-point SFA probe: isl=8192 q=256 nh=64 w8a8"
echo "[diagnose] watch for: is_sparse=True, attn_module=AscendSFAImpl,"
echo "           and whether the FIA self-test reports 561003 binary-not-found."

python3 "${SCRIPT_DIR}/collect_mla_module.py" \
  --mode context \
  --model "${MODEL_DIR}" \
  --quick \
  --seq-len 8192 \
  --batch-size 1 \
  --num-heads-override 64 \
  --chunk-query-len 256 \
  --quantization ascend \
  --output-dir ./_dsa_diag \
  --device "${DEVICE}" \
  --bench-iters 20 2>&1 | grep -iE "PROBE|is_sparse|SFA|FIA|561003|binary|AscendSFA|AscendMLA|FAILED|WARN|latency|b=1" || true

echo "[diagnose] DONE. If is_sparse=False or FIA shows 561003 -> SFA kernel"
echo "  not executing => synthetic collector cannot measure DSA on this box;"
echo "  fall back to profiler-derived modeling."
