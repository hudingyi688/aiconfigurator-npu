#!/usr/bin/env python3
"""Inspect vllm-ascend MoE dispatch+combine APIs on the actual NPU install.

Goal: dump the real schema of every class the upcoming collect_moe_dispatch_combine
collector needs to instantiate, so we can write the collector against the
exact signatures that exist on this vllm-ascend version (no source-code
guessing).

Usage on the NPU host (single card is enough; we are NOT calling any
collective op here, only introspecting Python classes):

    PYTHONPATH=collector python3 tools/inspect_moe_dispatch_api.py

Prints:
  - FusedMC2CommImpl __init__ signature + class methods
  - MoEFusedExpertsInput / MoEWeights / MoERouting / MoETopK fields
  - TokenDispatcherWithMC2 / PrepareAndFinalizeWithMC2 init signatures
  - The MoEConfig dataclass / NamedTuple fields
  - The dispatch_ffn_combine torch op schema (what scale layout / dtype is
    expected, what `group` looks like, what `expert_token_nums` is for)
  - Example: build a minimal MoEConfig and instantiate FusedMC2CommImpl
            without dist (skip the token_dispatcher init that needs HCCL)

Output: prints to stdout + saves /tmp/moe_dispatch_api_dump.txt for paste-back.
"""
from __future__ import annotations

import inspect
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

# Don't initialize torch_npu / dist here — we want pure introspection.
# The classes themselves should import fine from CPU-side.

OUT_PATH = "/tmp/moe_dispatch_api_dump.txt"


def _h(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _show_class(cls, name: str | None = None) -> None:
    name = name or cls.__name__
    print(f"\n  class {name}:")
    print(f"    module:  {cls.__module__}")
    print(f"    file:    {inspect.getsourcefile(cls)}")
    if hasattr(cls, "__init__"):
        try:
            print(f"    __init__: {inspect.signature(cls.__init__)}")
        except (TypeError, ValueError):
            pass
    # dataclass fields if any
    if hasattr(cls, "__dataclass_fields__"):
        print(f"    dataclass fields:")
        for fname, f in cls.__dataclass_fields__.items():
            print(f"      {fname:30s} : {f.type!r} = {f.default!r}")
    # NamedTuple fields
    if hasattr(cls, "_fields"):
        print(f"    NamedTuple fields: {cls._fields}")
    # public methods
    methods = [
        m for m in dir(cls)
        if not m.startswith("_") and callable(getattr(cls, m, None))
    ]
    if methods:
        print(f"    public methods: {methods[:20]}")


def _show_op_schema(op_path: str) -> None:
    print(f"\n  torch op: {op_path}")
    try:
        import torch
        ns_name, op_name = op_path.split(".")[-2], op_path.split(".")[-1]
        ns = getattr(torch.ops, ns_name)
        op = getattr(ns, op_name)
        for ov in op.overloads():
            schema = getattr(op, ov)._schema
            print(f"    overload {ov!r}: {schema}")
    except Exception as e:
        print(f"    (op not registered or load failed: {type(e).__name__}: {e})")


def _try(thunk, label: str) -> None:
    try:
        thunk()
    except Exception as e:
        print(f"\n  [{label}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def main() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run()
    out = buf.getvalue()
    print(out)
    with open(OUT_PATH, "w") as f:
        f.write(out)
    print(f"\n(also saved to {OUT_PATH})")


def _run() -> None:
    print(f"python    : {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")

    _h("[1] vllm-ascend version + install path")
    _try(_show_vllm_ascend, "vllm_ascend import")

    _h("[2] FusedMC2CommImpl + base class")
    _try(_show_fused_mc2, "moe_comm_method")

    _h("[3] MoEFusedExpertsInput / MoEWeights / MoERouting / MoETopK / FusedExpertsResult")
    _try(_show_moe_dataclasses, "MoE dataclasses")

    _h("[4] TokenDispatcherWithMC2 / PrepareAndFinalizeWithMC2")
    _try(_show_token_dispatcher, "token_dispatcher")

    _h("[5] MoEConfig (or whatever it's called)")
    _try(_show_moe_config, "moe_config")

    _h("[6] dispatch_ffn_combine torch op schema")
    _try(_show_dispatch_ops, "torch.ops._C_ascend")

    _h("[7] enable_fused_mc2 env var + related")
    _try(_show_envs, "vllm_ascend.envs")

    _h("[8] Try to instantiate FusedMC2CommImpl (no dist)")
    _try(_try_instantiate, "FusedMC2CommImpl()")


def _show_vllm_ascend() -> None:
    import vllm_ascend
    print(f"  vllm_ascend.__file__   : {vllm_ascend.__file__}")
    print(f"  vllm_ascend.__version__: {getattr(vllm_ascend, '__version__', '(missing)')}")


def _show_fused_mc2() -> None:
    from vllm_ascend.ops.fused_moe import moe_comm_method as m
    _show_class(m.FusedMC2CommImpl)
    if hasattr(m, "MoECommMethod"):
        _show_class(m.MoECommMethod, "MoECommMethod")
    # Source for fused_experts
    try:
        src = inspect.getsource(m.FusedMC2CommImpl.fused_experts)
        print(f"\n  --- FusedMC2CommImpl.fused_experts source ---")
        for line in src.splitlines()[:80]:
            print(f"    {line}")
    except Exception as e:
        print(f"    (getsource failed: {e})")


def _show_moe_dataclasses() -> None:
    candidates = [
        "vllm_ascend.ops.fused_moe.token_dispatcher",
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        "vllm_ascend.ops.fused_moe.fused_moe",
        "vllm_ascend.ops.fused_moe",
    ]
    found = {}
    for modname in candidates:
        try:
            import importlib
            mod = importlib.import_module(modname)
            for cls_name in (
                "MoEFusedExpertsInput", "MoEWeights", "MoERouting",
                "MoETopK", "FusedExpertsResult", "MoEConfig",
            ):
                if hasattr(mod, cls_name) and cls_name not in found:
                    found[cls_name] = (mod, getattr(mod, cls_name))
        except Exception as e:
            print(f"  import {modname}: {type(e).__name__}: {e}")
    for cls_name, (mod, cls) in found.items():
        print(f"\n  found {cls_name} in {mod.__name__}")
        _show_class(cls, cls_name)


def _show_token_dispatcher() -> None:
    from vllm_ascend.ops.fused_moe import moe_comm_method as m
    for cls_name in ("TokenDispatcherWithMC2", "PrepareAndFinalizeWithMC2"):
        if hasattr(m, cls_name):
            _show_class(getattr(m, cls_name), cls_name)
        else:
            print(f"  {cls_name} not in moe_comm_method, searching submodules...")
            try:
                from vllm_ascend.ops.fused_moe import token_dispatcher as td
                if hasattr(td, cls_name):
                    _show_class(getattr(td, cls_name), cls_name)
            except Exception as e:
                print(f"    {type(e).__name__}: {e}")


def _show_moe_config() -> None:
    candidates = [
        ("vllm_ascend.ops.fused_moe.fused_moe", "AscendFusedMoEConfig"),
        ("vllm_ascend.ops.fused_moe.fused_moe", "MoEConfig"),
        ("vllm_ascend.ops.fused_moe", "MoEConfig"),
        ("vllm.model_executor.layers.fused_moe.config", "FusedMoEConfig"),
        ("vllm.model_executor.layers.fused_moe.config", "MoEConfig"),
    ]
    for modname, cls_name in candidates:
        try:
            import importlib
            mod = importlib.import_module(modname)
            if hasattr(mod, cls_name):
                _show_class(getattr(mod, cls_name), f"{modname}.{cls_name}")
        except Exception as e:
            print(f"  {modname}.{cls_name}: {type(e).__name__}: {e}")


def _show_dispatch_ops() -> None:
    import torch
    # Need vllm_ascend_C.so loaded for _C_ascend ops to register
    try:
        import vllm_ascend
        from vllm_ascend.platform import NPUPlatform
        NPUPlatform.import_kernels()
        import glob
        root = os.path.dirname(vllm_ascend.__file__)
        for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
            torch.ops.load_library(so)
    except Exception as e:
        print(f"  (kernel load: {type(e).__name__}: {e})")

    for op_path in (
        "torch.ops._C_ascend.dispatch_ffn_combine",
        "torch.ops._C_ascend.dispatch_gmm_combine_decode",
        "torch.ops._C_ascend.npu_moe_distribute_dispatch_v2",
        "torch.ops._C_ascend.npu_moe_distribute_combine_v2",
        "torch.ops.npu.npu_moe_distribute_dispatch_v2",
        "torch.ops.npu.npu_moe_distribute_combine_v2",
    ):
        _show_op_schema(op_path)


def _show_envs() -> None:
    from vllm_ascend import envs
    interesting = [
        k for k in dir(envs)
        if "MC2" in k or "MOE" in k or "DISPATCH" in k
    ]
    print(f"  vllm_ascend.envs MC2/MOE/DISPATCH:")
    for k in interesting:
        try:
            v = getattr(envs, k)
            if callable(v):
                v = v()
        except Exception as e:
            v = f"(err: {e})"
        print(f"    {k} = {v!r}")
    print()
    print(f"  os.environ values:")
    for k in sorted(os.environ):
        if "MC2" in k or "MOE" in k or "VLLM_ASCEND" in k:
            print(f"    {k} = {os.environ[k]!r}")


def _try_instantiate() -> None:
    """Try to build FusedMC2CommImpl with the smallest plausible config,
    not yet calling fused_experts. This tells us which fields MoEConfig
    actually requires."""
    from vllm_ascend.ops.fused_moe import moe_comm_method as m

    # The class signature lookup tells us the parameter name; the
    # construction will fail if MoEConfig fields are missing — and that
    # failure tells us what we need to fill in.
    sig = inspect.signature(m.FusedMC2CommImpl.__init__)
    print(f"  FusedMC2CommImpl.__init__{sig}")
    print(f"  (full instantiation requires a real MoEConfig — listing what")
    print(f"   to populate based on the signature above)")


if __name__ == "__main__":
    main()
