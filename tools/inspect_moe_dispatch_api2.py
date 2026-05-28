#!/usr/bin/env python3
"""Round 2 inspection — pin down dataclass fields we missed first time.

Targets:
  - MoEWeights, MoERoutingParams, MoEQuantParams
    (referenced by MoEFusedExpertsInput from moe_stage_contracts)
  - FusedMoEParallelConfig (nested in FusedMoEConfig)
  - MoEActivation, RoutingMethodType (enums for FusedMoEConfig)
  - TokenDispatcherWithMC2.token_dispatch / token_combine signatures
    (since __init__ is **kwargs, the real contract is in the methods)
  - PrepareAndFinalizeWithMC2 deeper internals (group_name init flow)

Usage on NPU host (single card):
    PYTHONPATH=collector python3 tools/inspect_moe_dispatch_api2.py
"""
from __future__ import annotations

import inspect
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

OUT_PATH = "/tmp/moe_dispatch_api_dump_v2.txt"


def _h(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _show_class(cls, name=None):
    name = name or cls.__name__
    print(f"\n  class {name}:")
    print(f"    module:  {cls.__module__}")
    try:
        print(f"    file:    {inspect.getsourcefile(cls)}")
    except TypeError:
        pass
    if hasattr(cls, "__init__"):
        try:
            print(f"    __init__: {inspect.signature(cls.__init__)}")
        except (TypeError, ValueError):
            pass
    if hasattr(cls, "__dataclass_fields__"):
        print(f"    dataclass fields:")
        for fname, f in cls.__dataclass_fields__.items():
            default = f.default if not isinstance(f.default, type(inspect.Parameter.empty)) else "(required)"
            print(f"      {fname:32s} : {f.type!r:50s} default={default!r}")
    if hasattr(cls, "_fields"):
        print(f"    NamedTuple fields: {cls._fields}")
    if isinstance(cls, type) and issubclass(cls, type) is False:
        # check for Enum members
        try:
            from enum import Enum
            if issubclass(cls, Enum):
                print(f"    enum members: {[m.name for m in cls]}")
        except (TypeError, ImportError):
            pass
    methods = [
        m for m in dir(cls)
        if not m.startswith("_") and callable(getattr(cls, m, None))
    ]
    if methods:
        print(f"    public methods: {methods[:25]}")


def _show_method_source(cls, method_name, max_lines=60):
    print(f"\n  --- {cls.__name__}.{method_name} source ---")
    fn = getattr(cls, method_name, None)
    if fn is None:
        print(f"    (not found)")
        return
    try:
        print(f"    signature: {inspect.signature(fn)}")
        src = inspect.getsource(fn)
        for line in src.splitlines()[:max_lines]:
            print(f"    {line}")
    except (TypeError, OSError) as e:
        print(f"    (source unavailable: {e})")


def _try(thunk, label):
    try:
        thunk()
    except Exception as e:
        print(f"\n  [{label}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        _run()
    out = buf.getvalue()
    print(out)
    with open(OUT_PATH, "w") as f:
        f.write(out)
    print(f"\n(also saved to {OUT_PATH})")


def _run():
    _h("[1] moe_stage_contracts — MoEWeights / MoERoutingParams / MoEQuantParams")
    _try(_show_stage_contracts, "moe_stage_contracts")

    _h("[2] FusedMoEParallelConfig + enums")
    _try(_show_parallel_config, "FusedMoEParallelConfig")

    _h("[3] TokenDispatcherWithMC2 method signatures + token_dispatch source")
    _try(_show_token_dispatcher_methods, "token_dispatcher methods")

    _h("[4] PrepareAndFinalizeWithMC2.prepare source")
    _try(_show_prepare_finalize_source, "prepare/finalize source")

    _h("[5] FusedMC2CommImpl init flow (where token_dispatcher gets attached)")
    _try(_show_comm_init, "FusedMC2CommImpl init")

    _h("[6] How AscendFusedMoE wires up moe_all_to_all_group_name")
    _try(_show_alltoall_group_setup, "alltoall group")

    _h("[7] MoEActivation / RoutingMethodType enum members")
    _try(_show_enums, "enums")


def _show_stage_contracts():
    from vllm_ascend.ops.fused_moe import moe_stage_contracts as m
    print(f"  module file: {m.__file__}")
    classes = [
        c for c in dir(m)
        if not c.startswith("_") and isinstance(getattr(m, c), type)
    ]
    print(f"  exported classes: {classes}")
    for cls_name in (
        "MoEWeights", "MoERoutingParams", "MoEQuantParams",
        "MoEFusedExpertsInput", "MoEFusedExpertsOutput",
    ):
        if hasattr(m, cls_name):
            _show_class(getattr(m, cls_name), cls_name)


def _show_parallel_config():
    from vllm.model_executor.layers.fused_moe import config as c
    classes = [
        n for n in dir(c)
        if not n.startswith("_") and isinstance(getattr(c, n), type)
    ]
    print(f"  exported classes: {classes}")
    for cls_name in (
        "FusedMoEParallelConfig", "MoEActivation", "RoutingMethodType",
        "FusedMoEConfig",
    ):
        if hasattr(c, cls_name):
            _show_class(getattr(c, cls_name), cls_name)
    # Show a static-method or factory if present
    if hasattr(c.FusedMoEParallelConfig, "make"):
        _show_method_source(c.FusedMoEParallelConfig, "make")


def _show_token_dispatcher_methods():
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
    _show_class(TokenDispatcherWithMC2, "TokenDispatcherWithMC2")
    for m in ("token_dispatch", "token_combine",
              "get_dispatch_mc2_kwargs", "get_combine_mc_kwargs"):
        _show_method_source(TokenDispatcherWithMC2, m, max_lines=40)
    # Constructor source — to see what kwargs it expects
    _show_method_source(TokenDispatcherWithMC2, "__init__", max_lines=80)


def _show_prepare_finalize_source():
    from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMC2
    _show_class(PrepareAndFinalizeWithMC2, "PrepareAndFinalizeWithMC2")
    for m in ("__init__", "prepare", "finalize"):
        _show_method_source(PrepareAndFinalizeWithMC2, m, max_lines=50)


def _show_comm_init():
    from vllm_ascend.ops.fused_moe import moe_comm_method as m
    _show_method_source(m.FusedMC2CommImpl, "__init__", max_lines=40)
    _show_method_source(m.FusedMC2CommImpl, "_get_token_dispatcher", max_lines=20)
    _show_method_source(m.FusedMC2CommImpl, "_get_prepare_finalize", max_lines=20)
    if hasattr(m.MoECommMethod, "__init__"):
        _show_method_source(m.MoECommMethod, "__init__", max_lines=40)


def _show_alltoall_group_setup():
    """The group string is the key parameter dispatch_ffn_combine wants;
    find where it's constructed in vllm-ascend."""
    import subprocess
    sp = "/usr/local/python3.11.14/lib/python3.11/site-packages"
    out = subprocess.run(
        ["grep", "-rn",
         "moe_all_to_all_group_name",
         os.path.join(sp, "vllm_ascend")],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines()[:30]:
        rel = line.replace(sp + "/", "")
        print(f"  {rel}")


def _show_enums():
    candidates = [
        ("vllm.model_executor.layers.fused_moe.activation", "MoEActivation"),
        ("vllm.model_executor.layers.fused_moe.config", "MoEActivation"),
        ("vllm.model_executor.layers.fused_moe.config", "RoutingMethodType"),
    ]
    for modname, cls_name in candidates:
        try:
            import importlib
            mod = importlib.import_module(modname)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                print(f"\n  {modname}.{cls_name}:")
                from enum import Enum
                if isinstance(cls, type) and issubclass(cls, Enum):
                    for member in cls:
                        print(f"    {member.name} = {member.value!r}")
                else:
                    print(f"    (not an Enum: {cls!r})")
        except Exception as e:
            print(f"  {modname}.{cls_name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
