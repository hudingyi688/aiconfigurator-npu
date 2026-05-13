#!/usr/bin/env bash
# Diagnose why npu_sparse_flash_attention still segfaults on GLM-5 DSA
# benchmarking. Temporarily injects a print before the kernel call,
# runs one small case, captures the tensor stats, then restores the
# original file.
#
# Usage (on the NPU host):
#     chmod +x tools/probe_sfa_inputs.sh
#     tools/probe_sfa_inputs.sh
#
# Environment:
#     MODEL_DIR - path to the model (default: /a3_inference/itask/workdir/models/GLM-5-w8a8/)
#     SEQ_LEN   - query seq_len to test (default: 512)
#     SFA_FILE  - path to installed sfa_v1.py (auto-detected from vllm_ascend)

set -u

MODEL_DIR="${MODEL_DIR:-/a3_inference/itask/workdir/models/GLM-5-w8a8/}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH="${BATCH:-1}"
LOG="/tmp/aic_probe_s${SEQ_LEN}.log"
REPO_DIR="${REPO_DIR:-/mnt/sfs_turbo/hdy02488569/aiconfigurator-npu}"

# --- locate installed sfa_v1.py --------------------------------------------
if [ -z "${SFA_FILE:-}" ]; then
    SFA_FILE=$(python3 -c "import vllm_ascend, os; print(os.path.join(os.path.dirname(vllm_ascend.__file__), 'attention', 'sfa_v1.py'))" 2>/dev/null)
fi

if [ ! -f "$SFA_FILE" ]; then
    echo "ERROR: cannot locate installed sfa_v1.py (got: $SFA_FILE)" >&2
    exit 1
fi

BAK="/tmp/sfa_v1.py.$(date +%s).bak"
echo "=== probe_sfa_inputs.sh ==="
echo "sfa_v1.py  : $SFA_FILE"
echo "backup     : $BAK"
echo "model_dir  : $MODEL_DIR"
echo "seq_len    : $SEQ_LEN"
echo "batch      : $BATCH"
echo "log        : $LOG"
echo "repo       : $REPO_DIR"
echo

# --- backup & patch ---------------------------------------------------------
cp "$SFA_FILE" "$BAK" || { echo "backup failed"; exit 1; }

python3 <<PYEOF
f = "$SFA_FILE"
with open(f, 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

# Probe 1: just before npu_sparse_flash_attention (the original probe)
target_sfa = 'attn_output = torch.ops._C_ascend.npu_sparse_flash_attention('
# Probe 2: just before mla_preprocess (the most likely NaN source)
target_mlapo = 'torch.ops._C_ascend.mla_preprocess('

def stat_block(indent):
    return (
        indent + "import torch as _t\n"
        + indent + "def _st(name, x):\n"
        + indent + "    if x is None: return f'{name}=None'\n"
        + indent + "    if not hasattr(x, 'shape'): return f'{name}={x!r}'\n"
        + indent + "    xf = x.float() if x.is_floating_point() else x\n"
        + indent + "    return (f'{name} shape={tuple(x.shape)} dtype={x.dtype} '\n"
        + indent + "            f'min={float(xf.min()):.4e} max={float(xf.max()):.4e} '\n"
        + indent + "            f'mean={float(xf.float().mean()):.4e} '\n"
        + indent + "            f'nan={_t.isnan(xf.float()).any().item() if x.is_floating_point() else False} '\n"
        + indent + "            f'inf={_t.isinf(xf.float()).any().item() if x.is_floating_point() else False} '\n"
        + indent + "            f'contig={x.is_contiguous()}')\n"
    )

# Pass 1: insert MLAPO probe (must process BEFORE the SFA probe so line numbers stay valid)
inserted_mlapo = False
inserted_sfa = False
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if (not inserted_mlapo) and (target_mlapo in line):
        indent = line[: len(line) - len(line.lstrip())]
        probe = (
            stat_block(indent)
            + indent + "print('[MLAPO IN]', _st('hidden_states', hidden_states), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('wd_qkv', self.wd_qkv), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('deq_scale_qkv', self.deq_scale_qkv), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('gamma1', self.gamma1), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('beta1', self.beta1), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('wu_q', self.wu_q), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('qb_deq_scl', self.qb_deq_scl), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('gamma2', self.gamma2), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('cos', cos), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('sin', sin), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('W_UK_T', self.W_UK_T), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('quant_scale0', self.quant_scale0), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('quant_offset0', self.quant_offset0), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('quant_bias_qkv', self.quant_bias_qkv), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('quant_scale1', self.quant_scale1), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('quant_offset1', self.quant_offset1), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('qb_qt_bias', self.qb_qt_bias), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('ctkv_scale', self.ctkv_scale), flush=True)\n"
            + indent + "print('[MLAPO IN]', _st('q_nope_scale', self.q_nope_scale), flush=True)\n"
        )
        new_lines.append(probe)
        inserted_mlapo = True
    if (not inserted_sfa) and (target_sfa in line):
        indent = line[: len(line) - len(line.lstrip())]
        probe = (
            stat_block(indent)
            + indent + "print('[SFA DBG]', _st('ql_nope', ql_nope), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('q_pe', q_pe), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('kv', kv), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('topk_indices', topk_indices), flush=True)\n"
        )
        new_lines.append(probe)
        inserted_sfa = True
    new_lines.append(line)
    i += 1

if not inserted_mlapo:
    raise SystemExit("ERROR: mla_preprocess call site not found")
if not inserted_sfa:
    raise SystemExit("ERROR: npu_sparse_flash_attention call site not found")

with open(f, 'w', encoding='utf-8') as fp:
    fp.writelines(new_lines)

print(f"patched MLAPO + SFA probes")
PYEOF

rc=$?
if [ $rc -ne 0 ]; then
    echo "patch failed, restoring backup"
    cp "$BAK" "$SFA_FILE"
    exit $rc
fi

# --- run the collector ------------------------------------------------------
cd "$REPO_DIR" || { echo "cannot cd $REPO_DIR"; cp "$BAK" "$SFA_FILE"; exit 1; }

# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
unset VLLM_ASCEND_ENABLE_MLAPO
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

python -X faulthandler collector/npu/collect_mla_module.py \
    --mode context --quick \
    --batch-size "$BATCH" --seq-len "$SEQ_LEN" \
    --model "$MODEL_DIR" --output-dir ./data \
    > "$LOG" 2>&1
PY_EXIT=$?

# --- restore ALWAYS ---------------------------------------------------------
cp "$BAK" "$SFA_FILE"
echo "restored $SFA_FILE from $BAK"

# --- report -----------------------------------------------------------------
echo
echo "PY_EXIT: $PY_EXIT"
echo
echo "=== MLAPO IN lines (inputs to mla_preprocess) ==="
grep "MLAPO IN" "$LOG" || echo "(no MLAPO IN)"
echo
echo "=== SFA DBG lines (inputs to sparse_flash_attention) ==="
grep "SFA DBG" "$LOG" || echo "(no SFA DBG line printed — crashed before the probe)"
echo
echo "=== MLAPO DIAG lines ==="
grep "MLAPO DIAG" "$LOG" || echo "(no MLAPO DIAG)"
echo
echo "=== last 40 lines of log ==="
tail -40 "$LOG"

exit $PY_EXIT
