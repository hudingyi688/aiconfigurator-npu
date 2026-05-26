#!/usr/bin/env bash
# Diagnose how vllm-ascend's OOT (out-of-tree) SFA op bin gets
# registered with the ACL runtime — and why collect_mla_module fails
# with errno 561003 "binary bin not found" while a real GLM-5 inference
# run on the same box succeeds.
#
# Hypothesis:
#   The collector only does torch.ops.load_library(vllm_ascend_C*.so),
#   which lets the PyTorch dispatcher see the op schema, but does NOT
#   make the OOT op binary discoverable to ACL runtime. A real
#   `LLM.generate()` path goes through vllm_ascend.platform init +
#   register_ascend_customop() + (maybe) vllm_ascend.vllm_ascend_C
#   init_module(), which sets ASCEND_CUSTOM_OPP_PATH or equivalent.
#
# Goal of this script: gather enough info to reproduce that startup
# sequence inside the collector — without touching vllm-ascend at all.
#
# Usage on the NPU host:
#     bash tools/diag_sfa_registration.sh
#
# Output: /tmp/sfa_reg/*.log  +  /tmp/sfa_reg.tar.gz

set -u

OUT="/tmp/sfa_reg"
mkdir -p "${OUT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

echo "=============================================================="
echo "[1/6] Current ASCEND_*_PATH / OPP / CUSTOM_OPP env"
echo "=============================================================="
{
    env | grep -E "^(ASCEND|CANN|OPP|TOOLCHAIN|VLLM_ASCEND)" | sort
    echo "---"
    echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-(unset)}"
    echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-(unset)}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(unset)}"
} | tee "${OUT}/01_env.log"

echo
echo "=============================================================="
echo "[2/6] vllm-ascend layout: OOT bin / op_api / vendors / so"
echo "=============================================================="
python3 - <<'PY' | tee "${OUT}/02_vllm_ascend_layout.log"
import os, glob, vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
print("vllm_ascend root :", root)
ver = getattr(vllm_ascend, "__version__", "(no __version__)")
print("vllm_ascend ver  :", ver)
print()
print("=== top-level entries ===")
for name in sorted(os.listdir(root)):
    full = os.path.join(root, name)
    kind = "DIR " if os.path.isdir(full) else "FILE"
    print(f"  {kind} {name}")
print()
print("=== relevant subtrees / file patterns ===")
patterns = [
    "vllm_ascend_C*.so",
    "**/vendors/**",
    "**/op_api*",
    "**/custom_opp*",
    "**/op_impl/**",
    "**/op_proto/**",
    "**/built-in/**",
    "**/ascend910_93/**",
    "**/sparse_flash_attention*",
    "**/lightning_indexer*",
    "**/*.json",
]
for pat in patterns:
    hits = glob.glob(os.path.join(root, pat), recursive=True)
    if not hits:
        continue
    print(f"--- {pat} ({len(hits)} hits, showing up to 25) ---")
    for p in hits[:25]:
        rel = os.path.relpath(p, root)
        print(" ", rel)
PY

echo
echo "=============================================================="
echo "[3/6] vllm_ascend.platform / __init__ — find init / register hooks"
echo "=============================================================="
python3 - <<'PY' | tee "${OUT}/03_init_hooks.log"
import os, vllm_ascend, pkgutil, importlib

root = os.path.dirname(vllm_ascend.__file__)
print(f"=== top-level submodules of vllm_ascend ===")
for sub in pkgutil.iter_modules(vllm_ascend.__path__, "vllm_ascend."):
    print(" ", sub.name)
print()

names_of_interest = [
    "vllm_ascend.platform",
    "vllm_ascend.vllm_ascend_C",
    "vllm_ascend.utils",
    "vllm_ascend.ops",
    "vllm_ascend.attention",
    "vllm_ascend.attention.sfa_v1",
]
for name in names_of_interest:
    try:
        m = importlib.import_module(name)
        print(f"--- {name}  @  {getattr(m, '__file__', '(builtin)')} ---")
        attrs = [a for a in dir(m) if any(
            t in a.lower() for t in (
                "init", "register", "customop", "platform",
                "load_library", "ascend_c", "opp_path",
            )
        )]
        for a in sorted(set(attrs)):
            print(f"    {a}")
    except Exception as e:
        print(f"--- {name}: import failed: {type(e).__name__}: {e}")
    print()

# Also dump anything that looks like a bin discovery side-effect
print("=== greppable register / init / opp_path tokens ===")
import subprocess
out = subprocess.check_output(
    ["grep", "-rn",
     "-e", "register_ascend_customop",
     "-e", "register_op_package",
     "-e", "init_module",
     "-e", "ASCEND_CUSTOM_OPP_PATH",
     "-e", "ASCEND_OPP_PATH",
     "-e", "load_library",
     root],
    stderr=subprocess.STDOUT, text=True,
).splitlines()
for line in out[:80]:
    rel = line.replace(root + "/", "")
    print(" ", rel)
PY

echo
echo "=============================================================="
echo "[4/6] Trace what happens when we trigger the SAME init path the"
echo "      collector currently uses (load .so, no LLM, no platform)"
echo "=============================================================="
python3 -X faulthandler 2>&1 <<'PY' | tee "${OUT}/04_collector_init_trace.log"
import os, glob, traceback, torch
print("--- step 1: import torch_npu, set device, allow_internal_format ---")
try:
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)
    import torch_npu
    print("  ok")
except Exception as e:
    traceback.print_exc()

print("--- step 2: import vllm_ascend ---")
import vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
print(f"  root: {root}")

print("--- step 3: torch.ops.load_library(vllm_ascend_C*.so) ---")
for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
    try:
        torch.ops.load_library(so)
        print(f"  loaded {so}")
    except Exception as e:
        print(f"  FAILED {so}: {e}")

print("--- step 4: see if ACL knows about the OOT op now ---")
print("  ASCEND_CUSTOM_OPP_PATH =", os.environ.get("ASCEND_CUSTOM_OPP_PATH", "(unset)"))
print("  attrs visible on torch.ops._C_ascend:",
      [x for x in dir(torch.ops._C_ascend) if not x.startswith("_")][:30])

print("--- step 5: try the failing call (collector path) ---")
import torch
dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
T, N, R, D, NB, BS, SC = 32, 64, 64, 512, 4, 128, 2048
q = torch.randn(T, N, D, dtype=bf16, device=dev)
qr = torch.randn(T, N, R, dtype=bf16, device=dev)
kv = torch.randn(NB, BS, 1, D, dtype=bf16, device=dev)
kr = torch.randn(NB, BS, 1, R, dtype=bf16, device=dev)
si = torch.randint(0, NB*BS, (T, 1, SC), dtype=i32, device=dev)
bt = torch.arange(NB, dtype=i32, device=dev).unsqueeze(0)
aslq = torch.tensor([T], dtype=i32, device=dev)
aslk = torch.tensor([NB*BS], dtype=i32, device=dev)
try:
    torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q, key=kv, value=kv, sparse_indices=si,
        scale_value=1.0/(D**0.5), sparse_block_size=1,
        block_table=bt, actual_seq_lengths_query=aslq,
        actual_seq_lengths_kv=aslk, query_rope=qr, key_rope=kr,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
    )
    print("  collector path: OK")
except Exception as e:
    print(f"  collector path: FAIL {type(e).__name__}: {repr(e)[:300]}")
PY

echo
echo "=============================================================="
echo "[5/6] Trace what happens when we call register_ascend_customop()"
echo "      / vllm_ascend.platform init (the bits a real LLM hits)"
echo "=============================================================="
python3 -X faulthandler 2>&1 <<'PY' | tee "${OUT}/05_full_init_trace.log"
import os, glob, traceback, torch
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
import torch_npu
import vllm_ascend
root = os.path.dirname(vllm_ascend.__file__)
for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
    torch.ops.load_library(so)

candidates = [
    ("vllm_ascend.utils",        "register_ascend_customop"),
    ("vllm_ascend.platform",     "AscendPlatform"),
    ("vllm_ascend.platform",     "register_ascend_customop"),
    ("vllm_ascend",              "register_ascend_customop"),
    ("vllm_ascend.vllm_ascend_C", "init_module"),
]
print("--- attempting each known init hook ---")
for mod_name, fn_name in candidates:
    try:
        import importlib
        m = importlib.import_module(mod_name)
        fn = getattr(m, fn_name, None)
        if fn is None:
            print(f"  {mod_name}.{fn_name}: not present")
            continue
        print(f"  {mod_name}.{fn_name}: calling ...")
        if fn_name == "AscendPlatform":
            inst = fn()
            print(f"     instantiated {inst}")
            for meth in ("pre_register_and_update", "set_device",
                         "register_ascend_customop", "init"):
                if hasattr(inst, meth):
                    try:
                        getattr(inst, meth)()
                        print(f"     {meth}() ok")
                    except Exception as e:
                        print(f"     {meth}() FAIL: {type(e).__name__}: {e}")
        else:
            fn()
            print(f"     ok")
    except Exception as e:
        print(f"  {mod_name}.{fn_name}: import/call FAIL: {type(e).__name__}: {e}")
print()
print("--- env after init ---")
for k in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
    print(f"  {k}={os.environ.get(k, '(unset)')}")

print("--- retry the collector SFA call ---")
dev, bf16, i32 = "npu:0", torch.bfloat16, torch.int32
T, N, R, D, NB, BS, SC = 32, 64, 64, 512, 4, 128, 2048
q = torch.randn(T, N, D, dtype=bf16, device=dev)
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
    print(f"  retry: OK out={tuple(out[0].shape) if isinstance(out, (tuple, list)) else tuple(out.shape)}")
except Exception as e:
    print(f"  retry: FAIL {type(e).__name__}: {repr(e)[:300]}")
PY

echo
echo "=============================================================="
echo "[6/6] Trace a real LLM init (importtime) to see which modules"
echo "      vllm-ascend touches before the first SFA call"
echo "=============================================================="
python3 -X importtime -c "
import os
os.environ.setdefault('VLLM_USE_V1', '1')
import vllm_ascend
print('vllm_ascend imported OK')
" 2>&1 | grep -iE "ascend|opp|register|customop|sparse|init_module|platform" \
       | head -60 | tee "${OUT}/06_importtime.log"

echo
echo "=============================================================="
echo "All done. Packaging logs..."
tar czf /tmp/sfa_reg.tar.gz -C /tmp sfa_reg
echo "  Tarball: /tmp/sfa_reg.tar.gz"
echo "  Folder : ${OUT}/"
ls -la "${OUT}/"
