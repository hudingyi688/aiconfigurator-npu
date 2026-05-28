"""Factory for vllm-ascend MoE dispatch+combine bench setup.

Encapsulates everything the probe walked through:
  - NPU / vllm-ascend kernel init (OOT path, _C_ascend.so loading)
  - torch.distributed init (HCCL backend, world from torchrun env vars)
  - vllm config + ParallelConfig (so initialize_model_parallel runs)
  - vllm TP/PP groups (ensure_model_parallel_initialized)
  - vllm-ascend MC2 group (_MC2) — built directly from the ep ranks
  - FusedMoEConfig + FusedMoEParallelConfig
  - FusedMC2CommImpl construction (vllm-ascend's MoE comm interface)
  - MoEFusedExpertsInput tensor allocation (W8A8 list-of-tensors layout)

The collector stays thin: it sets a (num_tokens, ep_size, dtype) spec,
asks the factory for a forward callable + a context to keep alive, and
hands the callable to bench_engine.benchmark_npu().
"""
from __future__ import annotations

import glob
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable

import torch


# ============================================================
# 1. NPU / vllm-ascend bring-up (mirrors collect_mla_module preamble)
# ============================================================
def ensure_oot_custom_opp_path() -> None:
    """Prepend vllm-ascend's OOT vendor dir to ASCEND_CUSTOM_OPP_PATH."""
    if os.environ.get("AIC_SKIP_OOT_PATH_FIX") in {"1", "true", "TRUE"}:
        return
    try:
        import vllm_ascend  # noqa
    except ImportError:
        return
    oot = os.path.join(
        os.path.dirname(vllm_ascend.__file__),
        "_cann_ops_custom", "vendors", "vllm-ascend",
    )
    if not os.path.isdir(oot):
        return
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if oot in existing.split(":"):
        return
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{oot}:{existing}" if existing else oot
    )


def ensure_npu_kernels_loaded() -> None:
    """Force-load vllm_ascend_C.so + register OOT bins."""
    import torch
    torch.npu.config.allow_internal_format = True
    import torch_npu  # noqa
    import vllm_ascend
    from vllm_ascend.platform import NPUPlatform
    NPUPlatform.import_kernels()
    root = os.path.dirname(vllm_ascend.__file__)
    for so in sorted(glob.glob(os.path.join(root, "vllm_ascend_C*.so"))):
        try:
            torch.ops.load_library(so)
        except Exception as e:
            print(f"[WARN] failed to load {so}: {e}", flush=True)


# ============================================================
# 2. torch.distributed + vllm parallel state
# ============================================================
@dataclass(frozen=True)
class DistContext:
    """Bundle of distributed-state handles."""

    rank: int
    world_size: int
    local_rank: int
    ep_world_size: int
    ep_rank: int
    ep_group: torch.distributed.ProcessGroup
    group_name: str   # HCCL comm name string for the dispatch_ffn_combine op


def setup_distributed(ep_world_size: int) -> DistContext:
    """Init torch.distributed (via vllm) + create EP group + name string.

    Reads RANK / WORLD_SIZE / LOCAL_RANK from torchrun's env vars.
    Caller must run inside set_current_vllm_config(...) so vllm's
    initialize_model_parallel can read get_current_vllm_config().
    """
    import torch.distributed as dist
    from vllm.distributed import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size % ep_world_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be a multiple of "
            f"ep_world_size ({ep_world_size})"
        )

    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size, rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank, backend="hccl",
        )

    # vllm TP/PP groups (PrepareAndFinalizeWithMC2 ctor needs _TP).
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    # Build EP group covering ranks [floor(rank/ep_size)*ep_size, ...].
    # For sweep simplicity we keep ranks contiguous.
    ep_group_id = rank // ep_world_size
    ep_ranks = list(
        range(ep_group_id * ep_world_size, (ep_group_id + 1) * ep_world_size)
    )
    ep_group = dist.new_group(ranks=ep_ranks)

    # vllm-ascend MC2 group (manual stash — bypass init_ascend_model_parallel
    # which needs full ascend_config / vllm_config that we don't build).
    from vllm.distributed import init_model_parallel_group
    from vllm_ascend.distributed import parallel_state as ps
    if ps._MC2 is None:
        backend_name = dist.get_backend(ep_group)
        # Group spec is rows of ranks; for our flat sweep one row per
        # ep group is enough.
        all_ep_ranks = [
            list(range(g * ep_world_size, (g + 1) * ep_world_size))
            for g in range(world_size // ep_world_size)
        ]
        ps._MC2 = init_model_parallel_group(
            all_ep_ranks, local_rank, backend_name, group_name="mc2",
        )

    backend = ep_group._get_backend(torch.device("npu"))
    group_name = backend.get_hccl_comm_name(rank % ep_world_size)

    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        ep_world_size=ep_world_size,
        ep_rank=rank % ep_world_size,
        ep_group=ep_group,
        group_name=group_name,
    )


# ============================================================
# 3. vllm config (held for the full sweep)
# ============================================================
def make_vllm_config_context():
    """Return a context manager that puts a minimal VllmConfig in scope.

    Required by initialize_model_parallel and TokenDispatcherWithMC2.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    cfg = VllmConfig()
    try:
        cfg.parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
        )
    except Exception:
        pass
    return set_current_vllm_config(cfg)


# ============================================================
# 4. The bench spec + factory
# ============================================================
SUPPORTED_QUANT_TYPES = ("bf16", "w8a8_dynamic")


@dataclass(frozen=True)
class DispatchSpec:
    """One bench point."""
    num_tokens: int       # M (per-rank tokens)
    hidden: int           # H
    inter: int            # I (per-expert FFN intermediate)
    num_experts: int      # NE (logical experts, global)
    topk: int             # K
    ep_world_size: int    # EP
    quant_type: str       # 'bf16' | 'w8a8_dynamic'

    @property
    def num_local_experts(self) -> int:
        if self.num_experts % self.ep_world_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_world_size ({self.ep_world_size})"
            )
        return self.num_experts // self.ep_world_size


def build_moe_config(spec: DispatchSpec, dctx: DistContext, dtype: torch.dtype):
    """Build FusedMoEConfig matching the spec."""
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig,
        MoEActivation, RoutingMethodType,
    )
    parallel_cfg = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1,
        ep_size=spec.ep_world_size,
        tp_rank=0, pcp_rank=0, dp_rank=0,
        ep_rank=dctx.ep_rank,
        sp_size=1, use_ep=True,
        all2all_backend="naive", enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=spec.num_experts,
        experts_per_token=spec.topk,
        hidden_dim=spec.hidden,
        intermediate_size_per_partition=spec.inter,
        num_local_experts=spec.num_local_experts,
        num_logical_experts=spec.num_experts,
        activation=MoEActivation.SILU,
        device=torch.device(f"npu:{dctx.local_rank}"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_cfg,
        in_dtype=dtype,
    )


def _make_w8a8_inputs(spec: DispatchSpec, dctx: DistContext):
    """Build W8A8 weights/scales as list[Tensor] (one per local expert)."""
    import torch_npu
    dev = f"npu:{dctx.local_rank}"
    NL = spec.num_local_experts
    H, I = spec.hidden, spec.inter

    w1_stacked = torch.randint(
        -128, 127, (NL, H, 2 * I), dtype=torch.int8, device=dev,
    )
    w2_stacked = torch.randint(
        -128, 127, (NL, I, H), dtype=torch.int8, device=dev,
    )
    w1_scale_fp32 = torch.rand(NL, 2 * I, dtype=torch.float32, device=dev) * 0.1 + 0.01
    w2_scale_fp32 = torch.rand(NL, H,     dtype=torch.float32, device=dev) * 0.1 + 0.01

    # npu_trans_quant_param packs fp32 scale -> int64 deq_scale; only
    # accepts 1D scale, call per-expert.
    def _trans(per_expert_fp32):
        outs = []
        for i in range(per_expert_fp32.shape[0]):
            outs.append(
                torch_npu.npu_trans_quant_param(
                    per_expert_fp32[i].contiguous(), None,
                ).unsqueeze(0)
            )
        return torch.cat(outs, dim=0).to(torch.int64)

    s1_stacked = _trans(w1_scale_fp32)
    s2_stacked = _trans(w2_scale_fp32)

    return (
        list(w1_stacked.unbind(0)),
        list(w2_stacked.unbind(0)),
        list(s1_stacked.unbind(0)),
        list(s2_stacked.unbind(0)),
    )


def _make_bf16_inputs(spec: DispatchSpec, dctx: DistContext):
    """Build BF16 weights as list[Tensor] (no scales)."""
    dev = f"npu:{dctx.local_rank}"
    NL = spec.num_local_experts
    H, I = spec.hidden, spec.inter

    w1_stacked = torch.randn(NL, H, 2 * I, dtype=torch.bfloat16, device=dev)
    w2_stacked = torch.randn(NL, I, H,     dtype=torch.bfloat16, device=dev)
    return list(w1_stacked.unbind(0)), list(w2_stacked.unbind(0)), None, None


def create_dispatch_combine_func(
    spec: DispatchSpec, dctx: DistContext,
) -> tuple[Callable[[], None], dict]:
    """Build a forward callable that runs one comm.fused_experts() pass.

    Returns:
      forward_fn: zero-arg callable, runs one dispatch+ffn+combine
      meta: dict with the exit_stack to close after benchmarking
    """
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEFusedExpertsInput, MoEWeights,
    )
    from vllm_ascend.ops.fused_moe.moe_stage_params import (
        MoERoutingParams, MoEQuantParams,
    )

    if spec.quant_type == "w8a8_dynamic":
        dtype = torch.bfloat16
        w1_list, w2_list, s1_list, s2_list = _make_w8a8_inputs(spec, dctx)
        # MoEQuantParams default has QuantType.NONE; vllm-ascend's
        # FusedMC2CommImpl sees w1_scale != None and dispatches the
        # quant path. comm_quant_mode=2 mirrors production W8A8.
        from vllm_ascend.quantization.quant_config import QuantType
        try:
            quant_params = MoEQuantParams(
                quant_type=QuantType.W8A8, comm_quant_mode=2,
            )
        except Exception:
            quant_params = MoEQuantParams()
    elif spec.quant_type == "bf16":
        # FusedMC2CommImpl asserts w1_scale/w2_scale != None, i.e. it
        # only supports W8A8. BF16 in production goes through a
        # different path: TokenDispatcherWithMC2.token_dispatch ->
        # per-expert GEMM -> token_combine (three separate ops, see
        # profiler MoeDistributeDispatchV2 + GroupedMatmul +
        # MoeDistributeCombineV2). That requires a custom collector
        # we have not written yet; skip with a clear error so the
        # sweep driver records "no bf16 data" rather than misreporting.
        raise NotImplementedError(
            "bf16 dispatch+combine bench requires the unfused "
            "TokenDispatcherWithMC2 path (token_dispatch + expert "
            "GEMM + token_combine). Not implemented yet — only "
            "W8A8 (FusedMC2CommImpl) is supported in this collector."
        )
    else:
        raise ValueError(f"unsupported quant_type: {spec.quant_type}")

    moe_cfg = build_moe_config(spec, dctx, dtype)
    comm = FusedMC2CommImpl(moe_cfg)

    M = spec.num_tokens
    H = spec.hidden
    NL = spec.num_local_experts
    NE = spec.num_experts
    K = spec.topk
    dev = f"npu:{dctx.local_rank}"

    x = torch.randn(M, H, dtype=dtype, device=dev)
    topk_ids = torch.randint(0, NE, (M, K), dtype=torch.int32, device=dev)
    probs = torch.rand(M, K, dtype=torch.float32, device=dev)
    expert_map = torch.arange(NL, dtype=torch.int32, device=dev)

    weights = MoEWeights(
        w1=w1_list, w2=w2_list,
        w1_scale=s1_list, w2_scale=s2_list,
    )
    routing = MoERoutingParams(
        expert_map=expert_map,
        global_redundant_expert_num=0,
        mc2_mask=None,
        apply_router_weight_on_input=False,
    )
    input_ = MoEFusedExpertsInput(
        hidden_states=x, topk_weights=probs, topk_ids=topk_ids,
        weights=weights, routing=routing, quant=quant_params,
    )

    def forward_fn() -> None:
        comm.fused_experts(input_)

    # Dry run to surface init failures here, not in the timing loop.
    forward_fn()
    torch.npu.synchronize()

    return forward_fn, {"comm": comm, "input": input_}


# ============================================================
# 5. One-call setup for the collector entry point
# ============================================================
def setup_all(ep_world_size: int) -> tuple[DistContext, ExitStack]:
    """One-line bring-up: returns (DistContext, exit_stack).

    Caller must call exit_stack.close() at the end.
    """
    ensure_oot_custom_opp_path()
    ensure_npu_kernels_loaded()

    stack = ExitStack()
    stack.enter_context(make_vllm_config_context())
    dctx = setup_distributed(ep_world_size)
    return dctx, stack
