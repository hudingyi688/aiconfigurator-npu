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

print("_C_ascend ops sample:", [x for x in dir(torch.ops._C_ascend) if "sparse_flash" in x], flush=True)

# --- 2. build inputs with the SAME shapes AIConfigurator produces --------
dev = "npu:0"
bf16 = torch.bfloat16
i32 = torch.int32

num_tokens = 512
num_heads = 64
rope_dim = 64
kv_lora = 512
num_blocks = 32
block_size = 128
sparse_count = 2048

torch.manual_seed(0)
ql_nope = torch.empty(num_tokens, num_heads, kv_lora, dtype=bf16, device=dev).uniform_(-4, 4)
q_pe    = torch.empty(num_tokens, num_heads, rope_dim, dtype=bf16, device=dev).uniform_(-12, 12)
kv      = torch.empty(num_blocks, block_size, 1, kv_lora, dtype=bf16, device=dev).uniform_(-4, 4)
key_rope= torch.empty(num_blocks, block_size, 1, rope_dim, dtype=bf16, device=dev).uniform_(-1, 1)

# topk_indices: (T, 1, sparse_count), all in [0, num_blocks*block_size)
topk = torch.randint(0, num_blocks * block_size, (num_tokens, 1, sparse_count),
                     dtype=i32, device=dev)

block_table = torch.arange(num_blocks, dtype=i32, device=dev).unsqueeze(0)  # (1, num_blocks)

aslq = torch.tensor([num_tokens], dtype=i32, device=dev)
aslk = torch.tensor([num_blocks * block_size], dtype=i32, device=dev)

print("--- inputs ---", flush=True)
print(f"ql_nope {tuple(ql_nope.shape)} {ql_nope.dtype} contig={ql_nope.is_contiguous()}", flush=True)
print(f"q_pe    {tuple(q_pe.shape)} {q_pe.dtype} contig={q_pe.is_contiguous()}", flush=True)
print(f"kv      {tuple(kv.shape)} {kv.dtype} contig={kv.is_contiguous()}", flush=True)
print(f"key_rope{tuple(key_rope.shape)} {key_rope.dtype} contig={key_rope.is_contiguous()}", flush=True)
print(f"topk    {tuple(topk.shape)} {topk.dtype} min={topk.min().item()} max={topk.max().item()}", flush=True)
print(f"block_table {tuple(block_table.shape)} {block_table.dtype} max={block_table.max().item()}", flush=True)
print(f"aslq={aslq.tolist()} aslk={aslk.tolist()}", flush=True)

torch.npu.synchronize()
print("--- calling npu_sparse_flash_attention ---", flush=True)

try:
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=kv,
        value=kv,
        sparse_indices=topk,
        scale_value=1.0 / (kv_lora ** 0.5),
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=aslq,
        actual_seq_lengths_kv=aslk,
        query_rope=q_pe,
        key_rope=key_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print(f"[OK] out shape={tuple(out.shape)} dtype={out.dtype} "
          f"min={out.float().min().item()} max={out.float().max().item()}", flush=True)
except Exception as e:
    print(f"[ERR] {type(e).__name__}: {e}", flush=True)
    raise
PYEOF

rc=$?
echo
echo "PY_EXIT: $rc"
echo "=== log tail (60 lines) ==="
tail -60 "$LOG"

exit $rc
