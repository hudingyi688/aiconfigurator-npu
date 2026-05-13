#!/usr/bin/env bash
# Test if npu_sparse_flash_attention dispatches when query_rope/key_rope
# are NOT passed. The OPP binList only has variants where inputs[7,8]
# (query_rope, key_rope) are unmapped (dtype 0,2 = undefined). So we
# suspect runtime gave up matching because we were passing these two.
#
# Also tries with/without kv broadcast and sparse_mode variants.

set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export PYTHONFAULTHANDLER=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

LOG="/tmp/sfa_attr_sweep.log"

python -X faulthandler <<'PYEOF' > "$LOG" 2>&1
import sys, glob, os, torch, torch_npu, vllm_ascend

torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
for so in sorted(glob.glob(os.path.join(os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"))):
    try: torch.ops.load_library(so)
    except Exception: pass

def build(num_tokens=512, num_heads=64, rope_dim=64, kv_lora=512,
          num_blocks=32, block_size=128, sparse_count=2048,
          include_rope=True):
    dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
    torch.manual_seed(0)
    q     = torch.empty(num_tokens, num_heads, kv_lora, dtype=bf16, device=dev).uniform_(-1, 1)
    qrope = torch.empty(num_tokens, num_heads, rope_dim, dtype=bf16, device=dev).uniform_(-1, 1) if include_rope else None
    kv    = torch.empty(num_blocks, block_size, 1, kv_lora, dtype=bf16, device=dev).uniform_(-1, 1)
    krope = torch.empty(num_blocks, block_size, 1, rope_dim, dtype=bf16, device=dev).uniform_(-1, 1) if include_rope else None
    topk  = torch.randint(0, num_blocks*block_size, (num_tokens, 1, sparse_count), dtype=i32, device=dev)
    bt    = torch.arange(num_blocks, dtype=i32, device=dev).unsqueeze(0)
    aslq  = torch.tensor([num_tokens], dtype=i32, device=dev)
    aslk  = torch.tensor([num_blocks*block_size], dtype=i32, device=dev)
    return dict(query=q, key=kv, value=kv, sparse_indices=topk,
                scale_value=1.0/(kv_lora**0.5), sparse_block_size=1,
                block_table=bt, actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
                query_rope=qrope, key_rope=krope,
                layout_query="TND", layout_kv="PA_BSND", sparse_mode=3)

def try_call(label, kwargs):
    try:
        out = torch.ops._C_ascend.npu_sparse_flash_attention(**kwargs)
        torch.npu.synchronize()
        if isinstance(out, (tuple, list)):
            print(f"  {label}: OK (tuple len={len(out)})  out[0] shape={tuple(out[0].shape)}")
        else:
            print(f"  {label}: OK  shape={tuple(out.shape)}")
    except Exception as e:
        msg = str(e).split('\n')[0][:160]
        print(f"  {label}: FAIL  {type(e).__name__}: {msg}")

print("=== Hypothesis: query_rope/key_rope being passed prevents dispatch ===")
kw = build(include_rope=True); try_call("with rope", kw)
kw = build(include_rope=False); try_call("without rope", kw)

print()
print("=== Sweep sparse_mode ===")
for sm in [0, 1, 2, 3, 4]:
    kw = build(include_rope=False); kw["sparse_mode"] = sm
    try_call(f"sparse_mode={sm} no-rope", kw)

print()
print("=== Sweep layout_kv ===")
for lkv in ["PA_BSND", "BSND", "BSH", "TND", "PA_BNSD"]:
    kw = build(include_rope=False); kw["layout_kv"] = lkv
    try_call(f"layout_kv={lkv}", kw)

print()
print("=== With rope + try explicit None via different API path ===")
# some schemas reject None vs omitted kwargs differently; try positional
kw = build(include_rope=True)
# Also sweep attention_mode if the op accepts it:
for am in [0, 1, 2]:
    try:
        out = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=kw["query"], key=kw["key"], value=kw["value"],
            sparse_indices=kw["sparse_indices"],
            scale_value=kw["scale_value"], sparse_block_size=kw["sparse_block_size"],
            block_table=kw["block_table"],
            actual_seq_lengths_query=kw["actual_seq_lengths_query"],
            actual_seq_lengths_kv=kw["actual_seq_lengths_kv"],
            query_rope=kw["query_rope"], key_rope=kw["key_rope"],
            layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3,
        )
        torch.npu.synchronize()
        print(f"  attention_mode={am} (not passed): OK")
        break
    except Exception as e:
        print(f"  attention_mode={am}: sig doesn't accept it -> {type(e).__name__}: {str(e).split(chr(10))[0][:100]}")
        break
PYEOF

rc=$?
echo
echo "PY_EXIT: $rc"
echo "=== log ==="
cat "$LOG"
exit $rc
