#!/usr/bin/env bash
# Verify the fix: NPUPlatform.import_kernels() sets ASCEND_CUSTOM_OPP_PATH
# which makes ACL runtime find the OOT vllm-ascend SFA bin.
#
# Before: torch.ops._C_ascend.npu_sparse_flash_attention -> 561003 binary bin not found
# After : same call should succeed (or at least change error class).
#
# Usage on the NPU host:
#     bash tools/verify_sfa_fix.sh
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONFAULTHANDLER=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

LOG="/tmp/sfa_fix_verify.log"

python3 -X faulthandler 2>&1 <<'PYEOF' | tee "$LOG"
import os, glob, torch

print("--- step 1: import torch_npu, set device ---")
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
import torch_npu

print("--- step 2: load vllm_ascend_C.so ---")
import vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
    torch.ops.load_library(so)

print("--- step 3: BEFORE import_kernels ---")
print(f"  ASCEND_CUSTOM_OPP_PATH = {os.environ.get('ASCEND_CUSTOM_OPP_PATH', '(unset)')!r}")

print("--- step 4: call NPUPlatform.import_kernels() ---")
from vllm_ascend.platform import NPUPlatform
NPUPlatform.import_kernels()

print("--- step 5: AFTER import_kernels ---")
print(f"  ASCEND_CUSTOM_OPP_PATH = {os.environ.get('ASCEND_CUSTOM_OPP_PATH', '(unset)')!r}")

print("--- step 6: call npu_sparse_flash_attention (collector path) ---")
dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
T, N, R, D, NB, BS, SC = 32, 64, 64, 512, 4, 128, 2048
q  = torch.randn(T, N, D, dtype=bf16, device=dev)
qr = torch.randn(T, N, R, dtype=bf16, device=dev)
kv = torch.randn(NB, BS, 1, D, dtype=bf16, device=dev)
kr = torch.randn(NB, BS, 1, R, dtype=bf16, device=dev)
si = torch.randint(0, NB*BS, (T, 1, SC), dtype=i32, device=dev)
bt = torch.arange(NB, dtype=i32, device=dev).unsqueeze(0)
aslq = torch.tensor([T], dtype=i32, device=dev)
aslk = torch.tensor([NB*BS], dtype=i32, device=dev)
try:
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q, key=kv, value=kv, sparse_indices=si,
        scale_value=1.0/(D**0.5), sparse_block_size=1,
        block_table=bt, actual_seq_lengths_query=aslq,
        actual_seq_lengths_kv=aslk, query_rope=qr, key_rope=kr,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
    )
    torch.npu.synchronize()
    if isinstance(out, (tuple, list)):
        print(f"  RESULT: OK tuple len={len(out)} out[0]={tuple(out[0].shape)}")
    else:
        print(f"  RESULT: OK shape={tuple(out.shape)}")
except Exception as e:
    msg = repr(e)[:400]
    print(f"  RESULT: FAIL {type(e).__name__}: {msg}")

print()
print("--- step 7: also try the collector quick path end-to-end ---")
print("  (running collect_mla_module --quick — will use the new init path)")
PYEOF

rc=${PIPESTATUS[0]}
echo
echo "PY_EXIT: $rc"
echo
echo "If step 6 prints 'RESULT: OK', the fix is good — go run:"
echo "  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \\"
echo "    python collector/npu/collect_mla_module.py --mode context --quick \\"
echo "    --batch-size 4 --seq-len 2048 --output-dir ./data/glm5_dsa_module"
