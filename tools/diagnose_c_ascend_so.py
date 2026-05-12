#!/usr/bin/env python
"""Find out why torch.ops._C_ascend is nearly empty on this host:
  - Is the _C_ascend .so shipped in the installed vllm-ascend?
  - Does it fail to load, or is it simply not present (build skipped)?
  - Which pip package installed vllm-ascend, which version?
  - Does the local source tree (if any) contain the C++ sources for
    mla_preprocess / batch_matmul_transpose / npu_sparse_flash_attention?

Run on the NPU host:
    python tools/diagnose_c_ascend_so.py
"""
import glob
import os
import subprocess
import sys


def section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
    return r.stdout.rstrip() or r.stderr.rstrip()


def main():
    import vllm_ascend

    root = os.path.dirname(vllm_ascend.__file__)
    parent = os.path.dirname(root)

    section("installed vllm-ascend")
    print(f"  import path : {vllm_ascend.__file__}")
    print(f"  package root: {root}")
    print(f"  site-packages parent: {parent}")

    section("pip show vllm-ascend")
    print(run([sys.executable, "-m", "pip", "show", "-f", "vllm-ascend"]))

    section("candidate .so files inside vllm_ascend/")
    so_files = []
    for path in glob.glob(os.path.join(root, "**", "*.so"), recursive=True):
        so_files.append(path)
    for path in so_files:
        size = os.path.getsize(path)
        print(f"  {size:>10d}  {path}")
    if not so_files:
        print("  (no .so inside vllm_ascend/)")

    section("look for _C_ascend* anywhere on sys.path")
    for p in sys.path:
        if not p or not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            if name.startswith("_C_ascend") or name.startswith("vllm_ascend_C"):
                full = os.path.join(p, name)
                print(f"  {full}")

    section("grep '_C_ascend' in installed python files")
    # Just list which .py files still reference it (so we know what's broken)
    r = subprocess.run(
        ["grep", "-rnlE", r"torch\.ops\._C_ascend", root],
        capture_output=True, text=True, check=False,
    )
    print(r.stdout.rstrip() or "  (no matches)")

    section("torch ops after import vllm_ascend")
    import torch
    for ns in ("npu", "_C_ascend", "vllm_ascend_C"):
        try:
            obj = getattr(torch.ops, ns)
            ops = [x for x in dir(obj) if not x.startswith("_")]
            print(f"  {ns:15s}  count={len(ops)}  sample={ops[:8]}")
        except Exception as e:
            print(f"  {ns:15s}  err: {e}")

    section("ldd on any candidate .so (to see unmet deps)")
    for path in so_files:
        name = os.path.basename(path)
        if "_C_ascend" in name or "ascend_C" in name or "vllm" in name:
            print(f"--- ldd {path} ---")
            print(run(["ldd", path]))

    section("local source tree hints (looking for csrc/ that builds _C_ascend)")
    # Try common local source locations; extend as needed
    candidates = [
        "/mnt/sfs_turbo/hdy02488569/vllm-ascend",
        os.path.expanduser("~/vllm-ascend"),
        "/workspace/vllm-ascend",
    ]
    for src in candidates:
        if not os.path.isdir(src):
            continue
        print(f"--- {src} ---")
        for sub in ("csrc", "csrc/ops", "csrc/kernels"):
            p = os.path.join(src, sub)
            if os.path.isdir(p):
                print(f"  {p}:")
                for f in sorted(os.listdir(p))[:40]:
                    print(f"    {f}")
        r = subprocess.run(
            ["grep", "-rl", "mla_preprocess", os.path.join(src, "csrc")]
            if os.path.isdir(os.path.join(src, "csrc"))
            else ["true"],
            capture_output=True, text=True, check=False,
        )
        print("  csrc refs to mla_preprocess:", (r.stdout.strip() or "(none)"))

    section("CANN / python / torch versions")
    import torch_npu
    print(f"  python    : {sys.version.split()[0]}")
    print(f"  torch     : {torch.__version__}")
    print(f"  torch_npu : {torch_npu.__version__}")
    for p in ("/usr/local/Ascend/ascend-toolkit/latest/version.info",
              "/usr/local/Ascend/cann-8.5.0/version.info",
              "/usr/local/Ascend/version.info"):
        if os.path.isfile(p):
            with open(p) as f:
                print(f"  {p}:\n    " + f.read().rstrip().replace("\n", "\n    "))
            break


if __name__ == "__main__":
    main()
