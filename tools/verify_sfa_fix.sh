#!/usr/bin/env bash
# Verify the import_kernels() fix. Run after rebooting the python
# process from a clean shell so torch_npu init doesn't segfault on
# stale ACL state.
#
# Usage on the NPU host:
#     bash tools/verify_sfa_fix.sh
set -u

cd "$(dirname "$0")/.." || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PYTHONFAULTHANDLER=1
export PYTHONPATH="collector:${PYTHONPATH:-}"

LOG="/tmp/sfa_fix_verify.log"
: > "$LOG"

# Run each step in its own python process so a segfault inside torch_npu
# init does not abort the whole verification.

echo "==============================================================" | tee -a "$LOG"
echo "[1/3] sanity: torch_npu import + set_device + tensor add" | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
python3 -X faulthandler 2>&1 <<'PYEOF' | tee -a "$LOG"
import sys, traceback
print(f"  python : {sys.executable}")
try:
    import torch
    print(f"  torch  : {torch.__version__}")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    import torch_npu
    print(f"  torch_npu: {torch_npu.__version__}")
    x = torch.zeros(1, device="npu:0") + 1
    torch.npu.synchronize()
    print(f"  add: ok x={x.item()}")
except Exception as e:
    traceback.print_exc()
    print(f"  FAIL {type(e).__name__}: {e}")
PYEOF

echo | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
echo "[2/3] env / import_kernels() effect" | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
python3 -X faulthandler 2>&1 <<'PYEOF' | tee -a "$LOG"
import os, glob, traceback, torch
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
import torch_npu

print(f"  BEFORE  ASCEND_CUSTOM_OPP_PATH = {os.environ.get('ASCEND_CUSTOM_OPP_PATH', '(unset)')!r}")

import vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
    torch.ops.load_library(so)

try:
    from vllm_ascend.platform import NPUPlatform
    NPUPlatform.import_kernels()
    print(f"  import_kernels() ok")
except Exception as e:
    traceback.print_exc()
    print(f"  import_kernels FAIL {type(e).__name__}: {e}")

print(f"  AFTER   ASCEND_CUSTOM_OPP_PATH = {os.environ.get('ASCEND_CUSTOM_OPP_PATH', '(unset)')!r}")

# also check the dir actually exists on disk
custom = os.path.join(root, "_cann_ops_custom", "vendors", "vllm-ascend")
print(f"  OOT dir on disk        : {custom}")
print(f"  OOT dir exists         : {os.path.isdir(custom)}")
if os.path.isdir(custom):
    for sub in sorted(os.listdir(custom))[:10]:
        print(f"    contains: {sub}")
PYEOF

echo | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
echo "[3/3] _C_ascend.npu_sparse_flash_attention call" | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
python3 -X faulthandler 2>&1 <<'PYEOF' | tee -a "$LOG"
import os, glob, traceback, torch
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
import torch_npu
import vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
    torch.ops.load_library(so)
from vllm_ascend.platform import NPUPlatform
NPUPlatform.import_kernels()

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
print(f"  ASCEND_CUSTOM_OPP_PATH = {os.environ.get('ASCEND_CUSTOM_OPP_PATH', '(unset)')!r}")
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
PYEOF

echo
echo "Done. Full log: $LOG"
echo
echo "If step 3 prints 'RESULT: OK', run the collector:"
echo "  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \\"
echo "    python collector/npu/collect_mla_module.py --mode context --quick \\"
echo "    --batch-size 4 --seq-len 2048 --output-dir ./data/glm5_dsa_module"
