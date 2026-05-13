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
echo "[1/4] _C_ascend.npu_sparse_flash_attention (TND / PA_BSND)"
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
try:
    torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q, key=kv, value=kv, sparse_indices=si,
        scale_value=1.0/(D**0.5), sparse_block_size=1,
        block_table=bt,
        actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
        query_rope=qr, key_rope=kr,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
    )
    print("OK", flush=True)
except Exception as e:
    # print full message — do NOT split on newline
    print("EXCEPTION:", flush=True)
    print(repr(e), flush=True)
PYEOF

echo
echo "=============================================================="
echo "[2/4] torch.ops.npu.npu_sparse_flash_attention (BSND, the real one)"
echo "=============================================================="
python -X faulthandler 2>&1 <<'PYEOF'
import torch, torch_npu
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)

dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
B, S, N, D = 1, 512, 64, 512
R = 64
T_kv = 4096
SC = 2048

q  = torch.randn(B, S, N, D, dtype=bf16, device=dev)
qr = torch.randn(B, S, N, R, dtype=bf16, device=dev)
k  = torch.randn(B, T_kv, N, D, dtype=bf16, device=dev)
v  = torch.randn(B, T_kv, N, D, dtype=bf16, device=dev)
kr = torch.randn(B, T_kv, N, R, dtype=bf16, device=dev)
si = torch.randint(0, T_kv, (B, S, N, SC), dtype=i32, device=dev)
aslq = torch.tensor([S], dtype=i32, device=dev)
aslk = torch.tensor([T_kv], dtype=i32, device=dev)

print("--- sweep attention_mode with query_rope/key_rope ---", flush=True)
for am in [0, 1, 2]:
    try:
        out = torch.ops.npu.npu_sparse_flash_attention(
            q, k, v, si, 1.0/(D**0.5),
            actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
            query_rope=qr, key_rope=kr,
            sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
            sparse_mode=3, attention_mode=am,
        )
        torch.npu.synchronize()
        if isinstance(out, (tuple, list)):
            print(f"attention_mode={am}: OK tuple len={len(out)} out[0]={tuple(out[0].shape)}", flush=True)
        else:
            print(f"attention_mode={am}: OK shape={tuple(out.shape)}", flush=True)
    except Exception as e:
        print(f"attention_mode={am}: EXCEPTION:", flush=True)
        print("    " + repr(e)[:400], flush=True)

print()
print("--- sweep attention_mode WITHOUT query_rope/key_rope ---", flush=True)
for am in [0, 1, 2]:
    try:
        out = torch.ops.npu.npu_sparse_flash_attention(
            q, k, v, si, 1.0/(D**0.5),
            actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
            sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
            sparse_mode=3, attention_mode=am,
        )
        torch.npu.synchronize()
        if isinstance(out, (tuple, list)):
            print(f"attention_mode={am} no_rope: OK tuple len={len(out)} out[0]={tuple(out[0].shape)}", flush=True)
        else:
            print(f"attention_mode={am} no_rope: OK shape={tuple(out.shape)}", flush=True)
    except Exception as e:
        print(f"attention_mode={am} no_rope: EXCEPTION:", flush=True)
        print("    " + repr(e)[:400], flush=True)
PYEOF

echo
echo "=============================================================="
echo "[3/4] Find vllm-ascend tests that exercise SparseFlashAttention"
echo "=============================================================="
SP=/usr/local/python3.11.14/lib/python3.11/site-packages
find "$SP/vllm_ascend" -name "*.py" 2>/dev/null \
    | xargs grep -l "npu_sparse_flash_attention\|SparseFlashAttention" 2>/dev/null \
    | grep -i test 2>/dev/null \
    | head -10
echo "--- and any non-test call sites (vllm_ascend + torch_npu) ---"
grep -rn "npu_sparse_flash_attention" "$SP/vllm_ascend" "$SP/torch_npu" 2>/dev/null \
    | grep -v __pycache__ \
    | head -30

echo
echo "=============================================================="
echo "[4/4] Op signature inspection"
echo "=============================================================="
python <<'PYEOF'
import torch, torch_npu, inspect
fn = getattr(torch_npu, "npu_sparse_flash_attention", None)
print("torch_npu.npu_sparse_flash_attention:", fn)
if fn:
    try: print("  signature:", inspect.signature(fn))
    except Exception as e: print("  signature err:", e)

ns = torch.ops._C_ascend
print()
print("torch.ops._C_ascend visible attrs:")
print(" ", [x for x in dir(ns) if not x.startswith("_")])

ns2 = torch.ops.npu
ops = [x for x in dir(ns2) if "sparse_flash" in x.lower()]
print()
print("torch.ops.npu sparse_flash* attrs:")
print(" ", ops)
for o in ops:
    op = getattr(ns2, o)
    print(f"  {o}: {op}")
    try:
        for ov in op.overloads():
            print(f"    overload {ov!r}  schema = {getattr(op, ov)._schema}")
    except Exception as e:
        print("    overloads err:", e)
PYEOF
