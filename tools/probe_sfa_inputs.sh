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

target = 'attn_output = torch.ops._C_ascend.npu_sparse_flash_attention('
for i, l in enumerate(lines):
    if target in l:
        indent = l[: len(l) - len(l.lstrip())]
        probe = (
            indent
            + "import torch as _t\n"
            + indent + "def _st(name, x):\n"
            + indent + "    if x is None: return f'{name}=None'\n"
            + indent + "    if not hasattr(x, 'shape'): return f'{name}={x!r}'\n"
            + indent + "    xf = x.float()\n"
            + indent + "    return (f'{name} shape={tuple(x.shape)} dtype={x.dtype} '\n"
            + indent + "            f'min={xf.min().item():.4e} max={xf.max().item():.4e} '\n"
            + indent + "            f'mean={xf.mean().item():.4e} '\n"
            + indent + "            f'nan={_t.isnan(xf).any().item()} inf={_t.isinf(xf).any().item()} '\n"
            + indent + "            f'stride={x.stride()} contig={x.is_contiguous()}')\n"
            + indent + "print('[SFA DBG]', _st('ql_nope', ql_nope), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('q_pe', q_pe), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('kv', kv), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('key_rope', key_rope), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('topk_indices', topk_indices), flush=True)\n"
            + indent + "print('[SFA DBG]', _st('block_table', block_table), flush=True)\n"
            + indent + "print(f'[SFA DBG] aslq={actual_seq_lengths_query.tolist()} aslk={actual_seq_lengths_key.tolist()} '\n"
            + indent + "      f'aslq.dtype={actual_seq_lengths_query.dtype} aslk.dtype={actual_seq_lengths_key.dtype} '\n"
            + indent + "      f'scale={self.scale} sparse_block_size=1 sparse_mode=3 '\n"
            + indent + "      f'layout_q=TND layout_kv=PA_BSND', flush=True)\n"
        )
        lines.insert(i, probe)
        with open(f, 'w', encoding='utf-8') as fp:
            fp.writelines(lines)
        print(f"patched at line {i+1}")
        break
else:
    raise SystemExit("ERROR: target line not found in sfa_v1.py")
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
echo "=== SFA DBG lines ==="
grep "SFA DBG" "$LOG" || echo "(no SFA DBG line printed — crashed before the probe)"
echo
echo "=== MLAPO DIAG lines ==="
grep "MLAPO DIAG" "$LOG" || echo "(no MLAPO DIAG)"
echo
echo "=== last 40 lines of log ==="
tail -40 "$LOG"

exit $PY_EXIT
