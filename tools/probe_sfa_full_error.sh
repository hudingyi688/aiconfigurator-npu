#!/usr/bin/env bash
# Capture the FULL aclnnSparseFlashAttention error (including the
# [FUNC:][FILE:][LINE:] tail that previous probes truncated) and look
# for vllm-ascend's own tests / call sites that already exercise the op
# correctly.
#
# Usage on the NPU host:
#     tools/probe_sfa_full_error.sh

set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export PYTHONFAULTHANDLER=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

LOG="/tmp/sfa_full_err.log"

echo "=============================================================="
echo "[1/3] Single direct call — full ACL traceback"
echo "=============================================================="
python -X faulthandler 2>&1 <<'PYEOF' | tee "$LOG"
import torch, torch_npu, vllm_ascend, glob, os
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
    torch.ops.load_library(so)

dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
T, N, R, D = 512, 64, 64, 512
NB, BS, SC = 32, 128, 2048
q   = torch.randn(T, N, D, dtype=bf16, device=dev)
qr  = torch.randn(T, N, R, dtype=bf16, device=dev)
kv  = torch.randn(NB, BS, 1, D, dtype=bf16, device=dev)
kr  = torch.randn(NB, BS, 1, R, dtype=bf16, device=dev)
si  = torch.randint(0, NB*BS, (T, 1, SC), dtype=i32, device=dev)
bt  = torch.arange(NB, dtype=i32, device=dev).unsqueeze(0)
aslq = torch.tensor([T], dtype=i32, device=dev)
aslk = torch.tensor([NB*BS], dtype=i32, device=dev)

print("--- calling _C_ascend.npu_sparse_flash_attention ---", flush=True)
torch.ops._C_ascend.npu_sparse_flash_attention(
    query=q, key=kv, value=kv, sparse_indices=si,
    scale_value=1.0/(D**0.5), sparse_block_size=1,
    block_table=bt,
    actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
    query_rope=qr, key_rope=kr,
    layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
)
PYEOF
echo "exit code: $?"

echo
echo "=============================================================="
echo "[2/3] Find vllm-ascend tests that exercise SparseFlashAttention"
echo "=============================================================="
SP=/usr/local/python3.11.14/lib/python3.11/site-packages
find "$SP/vllm_ascend" -name "*.py" 2>/dev/null \
    | xargs grep -l "npu_sparse_flash_attention\|SparseFlashAttention" 2>/dev/null \
    | grep -i test 2>/dev/null \
    | head -10
echo "--- and any non-test call sites ---"
grep -rn "npu_sparse_flash_attention" "$SP/vllm_ascend" "$SP/torch_npu" 2>/dev/null \
    | grep -v __pycache__ \
    | head -30

echo
echo "=============================================================="
echo "[3/3] torch_npu.npu_sparse_flash_attention signature (if any)"
echo "=============================================================="
python <<'PYEOF'
import torch_npu, inspect
fn = getattr(torch_npu, "npu_sparse_flash_attention", None)
print("torch_npu.npu_sparse_flash_attention:", fn)
if fn:
    try:
        print("  signature:", inspect.signature(fn))
    except Exception as e:
        print("  signature: <unavailable> (", e, ")")
    try:
        print("  doc       :", (fn.__doc__ or "")[:1000])
    except Exception:
        pass

import torch
ns = torch.ops._C_ascend
ops = [x for x in dir(ns) if not x.startswith("_")]
print()
print("torch.ops._C_ascend visible attrs (count={}):".format(len(ops)))
print(" ", ops)

print()
op = ns.npu_sparse_flash_attention
print("op:", op)
print("dir:", [x for x in dir(op) if not x.startswith("_")][:30])
try:
    print("schema:", op.schemas)
except Exception:
    pass
try:
    overloads = op.overloads()
    for ov in overloads:
        print("  overload:", ov, " schema:", op.__getattr__(ov)._schema)
except Exception as e:
    print("  overload introspect err:", e)
PYEOF
