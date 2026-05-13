#!/usr/bin/env bash
# Minimal standalone reproducer for npu_sparse_flash_attention.
# Calls the kernel directly with the same shapes the AIConfigurator
# collector constructs, but without any of the model / MLAPO plumbing.
# If this segfaults too, the bug is in the kernel input contract itself
# (shape / dtype / layout / sparse_mode / topk). If it does NOT segfault,
# the bug is upstream (something AIConfigurator sets in kv_cache /
# topk_indices is subtly wrong).
#
# Usage (on the NPU host):
#     tools/probe_sfa_standalone.sh

set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export PYTHONFAULTHANDLER=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

LOG="/tmp/sfa_standalone.log"

.venv/bin/python 2>/dev/null 2>&1 || PY=python
PY=python

$PY -X faulthandler <<'PYEOF' > "$LOG" 2>&1
import sys, glob, os, torch

# --- 1. bring up NPU + load _C_ascend (same as collect_mla_module) -------
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)

import torch_npu
import vllm_ascend
for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
    try:
        torch.ops.load_library(so)
    except Exception as e:
        print(f"[WARN] load_library failed: {e}", flush=True)

# --- 2. sweep T values to find which shapes CANN has prebuilt binaries for
def try_call(num_tokens, num_heads=64, sparse_count=2048,
             num_blocks=None, block_size=128, kv_lora=512, rope_dim=64):
    if num_blocks is None:
        num_blocks = max(2, (max(num_tokens, sparse_count) + block_size - 1) // block_size)
    dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32

    torch.manual_seed(0)
    ql_nope = torch.empty(num_tokens, num_heads, kv_lora, dtype=bf16, device=dev).uniform_(-1, 1)
    q_pe    = torch.empty(num_tokens, num_heads, rope_dim, dtype=bf16, device=dev).uniform_(-1, 1)
    kv      = torch.empty(num_blocks, block_size, 1, kv_lora, dtype=bf16, device=dev).uniform_(-1, 1)
    key_rope= torch.empty(num_blocks, block_size, 1, rope_dim, dtype=bf16, device=dev).uniform_(-1, 1)
    topk = torch.randint(0, num_blocks * block_size, (num_tokens, 1, sparse_count),
                         dtype=i32, device=dev)
    block_table = torch.arange(num_blocks, dtype=i32, device=dev).unsqueeze(0)
    aslq = torch.tensor([num_tokens], dtype=i32, device=dev)
    aslk = torch.tensor([num_blocks * block_size], dtype=i32, device=dev)

    try:
        out = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope, key=kv, value=kv, sparse_indices=topk,
            scale_value=1.0 / (kv_lora ** 0.5),
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=aslq,
            actual_seq_lengths_kv=aslk,
            query_rope=q_pe, key_rope=key_rope,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
        )
        torch.npu.synchronize()
        return f"OK out={tuple(out.shape)} {out.dtype}"
    except Exception as e:
        msg = str(e).split("\n")[0][:120]
        return f"FAIL {type(e).__name__}: {msg}"

# Test shapes typical for prefill / decode + a few extreme small/large
test_T = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
print("--- sweeping T (num_tokens) ---", flush=True)
for t in test_T:
    print(f"T={t:5d}  num_heads=64 sparse_count=2048  ->  {try_call(t)}", flush=True)

# Also test sparse_count variants (kernel may only have certain values prebuilt)
print()
print("--- sweeping sparse_count at T=2048 ---", flush=True)
for sc in [128, 256, 512, 1024, 2048, 4096]:
    print(f"T=2048 sparse_count={sc:5d}  ->  {try_call(2048, sparse_count=sc)}", flush=True)

# And num_heads (DSA usually has 1, 8, 16, 32, 64, 128)
print()
print("--- sweeping num_heads at T=2048 sparse_count=2048 ---", flush=True)
for nh in [1, 8, 16, 32, 64, 128]:
    print(f"T=2048 num_heads={nh:3d}  ->  {try_call(2048, num_heads=nh)}", flush=True)
PYEOF

rc=$?
echo
echo "PY_EXIT: $rc"
echo "=== full log ==="
cat "$LOG"

exit $rc
