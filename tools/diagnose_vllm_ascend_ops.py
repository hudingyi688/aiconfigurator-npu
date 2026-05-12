#!/usr/bin/env python
"""Diagnose which custom ops are registered in the installed vllm-ascend,
and whether the installed sfa_v1.py still references them.

Run on the NPU host:
    python tools/diagnose_vllm_ascend_ops.py
"""
import os
import subprocess


def list_ops():
    import torch  # noqa: F401
    import torch_npu

    print("=" * 60)
    print("torch.ops.<namespace> op listing")
    print("=" * 60)
    for ns in ("npu", "vllm", "vllm_ascend_C", "ascend_C", "_C", "_C_ascend"):
        try:
            obj = getattr(torch.ops, ns)
            ops = [x for x in dir(obj) if not x.startswith("_")]
            mla_ops = [x for x in ops if "mla" in x.lower()]
            prep_ops = [x for x in ops if "preproc" in x.lower()]
            sfa_ops = [x for x in ops if "sparse_flash" in x.lower() or "lightning" in x.lower()]
            print(f"  ns={ns!r:22s}  total={len(ops):4d}  "
                  f"mla={mla_ops}  preproc={prep_ops}  sfa={sfa_ops}")
        except Exception as e:
            print(f"  ns={ns!r:22s}  ERR: {e}")

    print()
    print("=" * 60)
    print("torch_npu.* candidate APIs")
    print("=" * 60)
    cands = [x for x in dir(torch_npu)
             if ("mla" in x.lower() or "preproc" in x.lower()
                 or "sparse_flash" in x.lower() or "lightning" in x.lower())]
    print(f"  {cands}")


def inspect_vllm_ascend():
    import vllm_ascend

    print()
    print("=" * 60)
    print("vllm-ascend installation")
    print("=" * 60)
    root = os.path.dirname(vllm_ascend.__file__)
    print(f"  path   : {root}")
    ver = getattr(vllm_ascend, "__version__", None)
    print(f"  version: {ver}")

    sfa = os.path.join(root, "attention", "sfa_v1.py")
    mla = os.path.join(root, "attention", "mla_v1.py")
    for label, path in (("sfa_v1.py", sfa), ("mla_v1.py", mla)):
        print()
        print(f"--- {label} @ {path} ---")
        if not os.path.isfile(path):
            print(f"  (file not found)")
            continue
        try:
            r = subprocess.run(
                ["grep", "-nE", r"_C_ascend\.(mla_preprocess|batch_matmul_transpose|npu_lightning_indexer|npu_sparse_flash_attention)", path],
                capture_output=True, text=True, check=False,
            )
            out = r.stdout.strip() or "(no _C_ascend.* sensitive refs)"
            print(out)
        except Exception as e:
            print(f"  grep failed: {e}")


def print_cann_torch_npu():
    print()
    print("=" * 60)
    print("CANN / torch_npu versions")
    print("=" * 60)
    for p in ("/usr/local/Ascend/ascend-toolkit/latest/version.info",
              "/usr/local/Ascend/version.info"):
        if os.path.isfile(p):
            print(f"--- {p} ---")
            with open(p) as f:
                print(f.read().rstrip())
            break
    else:
        print("  CANN version.info not found in standard paths")

    try:
        import torch_npu
        print(f"  torch_npu.__version__ = {torch_npu.__version__}")
    except Exception as e:
        print(f"  torch_npu version ERR: {e}")


def main():
    print_cann_torch_npu()
    list_ops()
    inspect_vllm_ascend()


if __name__ == "__main__":
    main()
