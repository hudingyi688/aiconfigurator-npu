#!/usr/bin/env python3
"""MVP probe for torch.ops._C_ascend.dispatch_ffn_combine on a single NPU.

Goal: confirm we can build all the inputs the op expects and call it
with world_size=1 (degenerate ep group, no real alltoall traffic).
This is NOT a benchmark — just a "does the call go through".

What this validates:
  1. weight1/2 and scale1/2 — Tensor[] list-of-tensors layout, one
     entry per local expert
  2. expert_idx / topk_ids dtype + shape (int32 vs int64)
  3. probs dtype (FP32) and layout (M, topk)
  4. group string from a single-rank HCCL group
  5. out / expert_token_nums output buffer shapes
  6. The path through FusedMC2CommImpl.fused_experts() if we want to
     stay one layer up (it requires _EXTRA_CTX.mc2_mask etc.)

Result: prints what works, what doesn't. No CSV output.

Usage on the NPU host (single card):
    PYTHONPATH=collector \\
    VLLM_ASCEND_ENABLE_FUSED_MC2=1 \\
    python3 tools/probe_dispatch_ffn_combine.py
"""
from __future__ import annotations

import os
import sys
import traceback


def _h(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str, e: Exception | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if e is not None:
        print(f"         {type(e).__name__}: {e}")


def main() -> None:
    # 1. NPU + vllm-ascend init (mirror collect_mla_module preamble)
    _h("[1] NPU + vllm-ascend init")
    try:
        import vllm_ascend  # noqa
        oot = os.path.join(
            os.path.dirname(vllm_ascend.__file__),
            "_cann_ops_custom", "vendors", "vllm-ascend",
        )
        if os.path.isdir(oot):
            existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            if oot not in existing.split(":"):
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                    f"{oot}:{existing}" if existing else oot
                )
        import torch
        torch.npu.config.allow_internal_format = True
        torch.npu.set_device(0)
        import torch_npu  # noqa
        from vllm_ascend.platform import NPUPlatform
        NPUPlatform.import_kernels()

        # Load _C_ascend ops
        import glob
        for so in sorted(glob.glob(os.path.join(
            os.path.dirname(vllm_ascend.__file__), "vllm_ascend_C*.so"
        ))):
            torch.ops.load_library(so)
        _ok("env + kernels loaded")
    except Exception as e:
        _fail("init", e)
        traceback.print_exc()
        return

    # 2. Single-rank torch.distributed (we still need a real HCCL group
    # because the op takes a group string).
    _h("[2] dist init with world=1")
    try:
        import torch.distributed as dist
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl", world_size=1, rank=0)
        ep_group = dist.new_group(ranks=[0])
        _ok(f"dist initialized, ep_group rank={dist.get_rank(group=ep_group)} "
            f"world={dist.get_world_size(group=ep_group)}")
    except Exception as e:
        _fail("dist init", e)
        traceback.print_exc()
        return

    # 3. Get the HCCL group name (this is the str the op wants)
    _h("[3] HCCL group name")
    try:
        backend = ep_group._get_backend(torch.device("npu"))
        group_name = backend.get_hccl_comm_name(0)
        _ok(f"group_name = {group_name!r}")
    except Exception as e:
        _fail("get_hccl_comm_name", e)
        traceback.print_exc()
        return

    # 4. Construct minimal inputs and call dispatch_ffn_combine directly
    _h("[4] direct call to torch.ops._C_ascend.dispatch_ffn_combine (W8A8)")
    try:
        # GLM-5-shaped tiny case for first call
        H = 6144                # hidden
        I = 2048                # moe_intermediate
        NE = 16                 # num_experts (small for probe)
        TOPK = 4
        NL = NE                 # num_local_experts (ep_size=1 -> all on this rank)
        M = 8                   # num_tokens

        dev, bf16, i8, i32, f32 = "npu:0", torch.bfloat16, torch.int8, torch.int32, torch.float32

        x = torch.randn(M, H, dtype=bf16, device=dev)
        # Per profiler: w1 = (NL, H, 2*I) when fused gate_up; w2 = (NL, I, H)
        # scale1, scale2: per-channel per-expert
        # Try the stacked-tensor layout first; fall back to list-of-tensors if rejected.
        w1 = torch.randint(-128, 127, (NL, H, 2 * I), dtype=i8, device=dev)
        w2 = torch.randint(-128, 127, (NL, I, H), dtype=i8, device=dev)
        s1 = torch.rand(NL, 2 * I, dtype=bf16, device=dev) * 0.1 + 0.01
        s2 = torch.rand(NL, H, dtype=bf16, device=dev) * 0.1 + 0.01

        topk_ids = torch.randint(0, NE, (M, TOPK), dtype=i32, device=dev)
        probs = torch.rand(M, TOPK, dtype=f32, device=dev)

        out = torch.empty_like(x)
        expert_token_nums = torch.zeros(NL, dtype=i32, device=dev)

        # The op signature says weight1/2/scale1/2 are Tensor[].
        # Try with stacked tensors wrapped into a single-element list:
        for variant in ("stacked-as-list", "list-of-experts"):
            print(f"\n  -- trying weight layout: {variant} --")
            if variant == "stacked-as-list":
                w1_arg, w2_arg = [w1], [w2]
                s1_arg, s2_arg = [s1], [s2]
            else:
                w1_arg = list(w1.unbind(0))
                w2_arg = list(w2.unbind(0))
                s1_arg = list(s1.unbind(0))
                s2_arg = list(s2.unbind(0))

            try:
                torch.ops._C_ascend.dispatch_ffn_combine(
                    x=x,
                    weight1=w1_arg,
                    weight2=w2_arg,
                    expert_idx=topk_ids,
                    scale1=s1_arg,
                    scale2=s2_arg,
                    probs=probs,
                    group=group_name,
                    max_output_size=65536,
                    out=out,
                    expert_token_nums=expert_token_nums,
                )
                torch.npu.synchronize()
                _ok(f"variant {variant!r}: out shape={tuple(out.shape)} "
                    f"expert_tokens={expert_token_nums[:NL].tolist()}")
                # Successful — record this layout
                print(f"\n  >>> WINNER: {variant!r}")
                break
            except Exception as e:
                _fail(f"variant {variant!r}", e)
    except Exception as e:
        _fail("setup tensors", e)
        traceback.print_exc()

    # 5. Try going through FusedMC2CommImpl.fused_experts() (one layer up)
    _h("[5] FusedMC2CommImpl.fused_experts() path (vllm-ascend interface layer)")
    try:
        from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
        from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
            MoEFusedExpertsInput, MoEWeights,
        )
        from vllm_ascend.ops.fused_moe.moe_stage_params import (
            MoERoutingParams, MoEQuantParams,
        )
        # MoEConfig requires FusedMoEParallelConfig — try minimal
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig, FusedMoEParallelConfig,
            MoEActivation, RoutingMethodType,
        )

        parallel_cfg = FusedMoEParallelConfig(
            tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
            tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
            sp_size=1, use_ep=True,
            all2all_backend="naive", enable_eplb=False,
        )
        moe_cfg = FusedMoEConfig(
            num_experts=NE,
            experts_per_token=TOPK,
            hidden_dim=H,
            intermediate_size_per_partition=I,
            num_local_experts=NL,
            num_logical_experts=NE,
            activation=MoEActivation.SILU,
            device=torch.device(dev),
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=parallel_cfg,
            in_dtype=bf16,
        )
        _ok(f"FusedMoEConfig built: {moe_cfg.num_experts=} {moe_cfg.hidden_dim=}")

        # FusedMC2CommImpl() touches get_mc2_group() which needs
        # vllm-ascend's distributed setup; this is the part we expect
        # to need a vllm_config. Probe the failure mode:
        try:
            comm = FusedMC2CommImpl(moe_cfg)
            _ok(f"FusedMC2CommImpl built")
            # If we got here, try fused_experts()
            input_ = MoEFusedExpertsInput(
                hidden_states=x,
                topk_weights=probs,
                topk_ids=topk_ids,
                weights=MoEWeights(w1=w1, w2=w2, w1_scale=s1, w2_scale=s2),
                routing=MoERoutingParams(
                    expert_map=torch.arange(NL, dtype=i32, device=dev),
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                quant=MoEQuantParams(),
            )
            try:
                result = comm.fused_experts(input_)
                torch.npu.synchronize()
                _ok(f"fused_experts ok: {tuple(result.routed_out.shape)}")
            except Exception as e:
                _fail("fused_experts call", e)
                traceback.print_exc()
        except Exception as e:
            _fail("FusedMC2CommImpl ctor", e)
            traceback.print_exc()

    except Exception as e:
        _fail("imports for path 5", e)
        traceback.print_exc()

    print()
    print("Done. The first variant that printed >>> WINNER above is the")
    print("weight layout to use in the real collector. If neither worked,")
    print("the 'TypeError' / 'EZxxx' errors will tell us what's wrong.")


if __name__ == "__main__":
    main()
