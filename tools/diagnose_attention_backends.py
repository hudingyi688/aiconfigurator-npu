#!/usr/bin/env python
"""Find which attention backend the installed vllm-ascend actually dispatches
to for a DeepSeek/GLM-style MLA config — and which internal branch runs.

Approach:
  1) List files in vllm_ascend/attention/ and grep for backend registration
     points (e.g. AttentionBackend classes, get_impl_cls, name='...').
  2) List the MLA-related Python entrypoints that do NOT call _C_ascend.*.
  3) Show the decision trees inside mla_v1.py and sfa_v1.py that pick
     the MLAPO / non-MLAPO branch, so we can see what config flags would
     route a GLM-5 run into a _C_ascend-free path.

Run on the NPU host:
    python tools/diagnose_attention_backends.py
"""
import os
import re
import subprocess


def section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def grep(path, pattern, flags=""):
    if not os.path.isfile(path):
        return f"(missing: {path})"
    r = subprocess.run(
        ["grep", "-nE"] + ([flags] if flags else []) + [pattern, path],
        capture_output=True, text=True, check=False,
    )
    return r.stdout.rstrip() or "(no match)"


def main():
    import vllm_ascend

    root = os.path.dirname(vllm_ascend.__file__)
    attn_dir = os.path.join(root, "attention")

    section("attention/ directory contents")
    for name in sorted(os.listdir(attn_dir)):
        print(" ", name)

    section("backend class / name / registration sites")
    for fname in sorted(os.listdir(attn_dir)):
        if not fname.endswith(".py"):
            continue
        p = os.path.join(attn_dir, fname)
        out = grep(p, r"class .*Backend|NAME *=|get_impl_cls|register|Attention\w*Backend")
        if out and out != "(no match)":
            print(f"--- {fname} ---")
            print(out)

    section("sfa_v1.py: branches leading to _C_ascend.mla_preprocess")
    sfa = os.path.join(attn_dir, "sfa_v1.py")
    # Print ~30 lines of context around the mla_preprocess call
    if os.path.isfile(sfa):
        r = subprocess.run(
            ["grep", "-nE", r"mla_preprocess|enable_mlapo|_sfa_preprocess|VLLM_ASCEND_ENABLE_MLAPO|num_input_tokens", sfa],
            capture_output=True, text=True, check=False,
        )
        print(r.stdout.rstrip() or "(no match)")

    section("mla_v1.py: branches leading to _C_ascend.mla_preprocess")
    mla = os.path.join(attn_dir, "mla_v1.py")
    if os.path.isfile(mla):
        r = subprocess.run(
            ["grep", "-nE", r"mla_preprocess|enable_mlapo|VLLM_ASCEND_ENABLE_MLAPO|num_input_tokens|npu_mla_prolog|paged_attention_mla", mla],
            capture_output=True, text=True, check=False,
        )
        print(r.stdout.rstrip() or "(no match)")

    section("Other MLA-capable modules that might NOT use _C_ascend")
    candidates = []
    for fname in sorted(os.listdir(attn_dir)):
        if not fname.endswith(".py"):
            continue
        p = os.path.join(attn_dir, fname)
        with open(p, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        uses_c_ascend = "_C_ascend" in txt
        uses_mla_prolog = "npu_mla_prolog" in txt or "paged_attention_mla" in txt
        if uses_mla_prolog:
            candidates.append((fname, uses_c_ascend, uses_mla_prolog))
    if candidates:
        print("  file                                uses_C_ascend  uses_mla_prolog")
        for f, a, b in candidates:
            print(f"  {f:35s}  {str(a):13s}  {b}")
    else:
        print("  (no file references npu_mla_prolog / paged_attention_mla)")

    section("Environment flags that influence backend selection")
    r = subprocess.run(
        ["grep", "-rnE", r"VLLM_ASCEND_\w+|ASCEND_ATTN|ATTN_BACKEND|attn_backend",
         os.path.join(root, "envs.py")],
        capture_output=True, text=True, check=False,
    )
    print(r.stdout.rstrip() or "(no match in envs.py)")

    section("Currently-set VLLM_* / ASCEND_* env vars")
    for k, v in sorted(os.environ.items()):
        if k.startswith("VLLM_") or k.startswith("ASCEND") or "ATTN" in k:
            print(f"  {k}={v}")


if __name__ == "__main__":
    main()
