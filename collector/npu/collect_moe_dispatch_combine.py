#!/usr/bin/env python3
"""MoE dispatch+combine collector for Ascend NPU.

Calls torch.ops._C_ascend.dispatch_ffn_combine via vllm-ascend's
FusedMC2CommImpl wrapper (W8A8 path) and torch_npu.npu_moe_distribute_*
via TokenDispatcherWithMC2 (BF16 path), measuring the end-to-end
dispatch + expert FFN + combine latency.

Launch (one ep_size at a time, see run_moe_dispatch_sweep.sh):

    VLLM_ASCEND_ENABLE_FUSED_MC2=1 \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=collector \\
    torchrun --nproc_per_node=8 \\
        collector/npu/collect_moe_dispatch_combine.py \\
            --hidden 6144 --inter 2048 --num-experts 256 --topk 8 \\
            --ep-size 8 \\
            --num-tokens-list 1 4 16 64 256 1024 4096 \\
            --quant-types bf16 w8a8_dynamic \\
            --output-dir ./data/moe_dispatch_combine

Output (rank-0 only):
    moe_dispatch_combine_perf.csv
        op_name, kernel_source, num_tokens, hidden, inter,
        num_experts, topk, ep_size, num_local_experts, dtype, latency_us
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_engine import benchmark_npu  # noqa: E402

from npu.moe_dispatch_factory import (  # noqa: E402
    DispatchSpec,
    SUPPORTED_QUANT_TYPES,
    create_dispatch_combine_func,
    setup_all,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=6144)
    p.add_argument("--inter", type=int, default=2048)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--ep-size", type=int, required=True,
                   help="EP world size; must divide torchrun's WORLD_SIZE.")
    p.add_argument("--num-tokens-list", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--quant-types", nargs="+", default=["bf16", "w8a8_dynamic"],
                   choices=list(SUPPORTED_QUANT_TYPES))
    p.add_argument("--output-dir", type=str, default="./data/moe_dispatch_combine")
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--num-runs", type=int, default=50)
    p.add_argument("--append", action="store_true",
                   help="Append to an existing per-ep CSV instead of "
                        "overwriting it. Default is to truncate and rewrite, "
                        "so each run produces a single-version, dup-free file. "
                        "Only use --append to resume a deliberately split run.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _csv_path(out_dir: Path, ep_size: int) -> Path:
    return out_dir / f"moe_dispatch_combine_ep{ep_size}.csv"


def _write_row(writer, spec: DispatchSpec, latency_us: float) -> None:
    writer.writerow([
        "moe_dispatch_combine",
        "vllm_ascend_FusedMC2",
        spec.num_tokens, spec.hidden, spec.inter,
        spec.num_experts, spec.topk, spec.ep_world_size,
        spec.num_local_experts, spec.quant_type,
        f"{latency_us:.4f}",
    ])


# Device-level error signatures that leave the NPU unrecoverable for the
# rest of the process: once seen, every subsequent op (including
# empty_cache / synchronize) re-raises, so the sweep must stop, not retry.
_FATAL_DEVICE_ERROR_MARKERS = (
    "507057",                 # HCCL/runtime suspect-remote error code
    "SUSPECT REMOTE ERROR",
    "PTA call acl api failed",
    "npuSynchronizeDevice",
)


def _is_fatal_device_error(exc: BaseException) -> bool:
    """True if exc looks like an unrecoverable NPU device/HCCL error."""
    text = str(exc)
    return any(marker in text for marker in _FATAL_DEVICE_ERROR_MARKERS)


def _ensure_hccl_buffsize(num_tokens_list, hidden: int, topk: int) -> None:
    """Set HCCL_BUFFSIZE (MB) large enough for the largest M in the sweep.

    dispatch_ffn_combine validates HCCL_BUFFSIZE against a hard formula
    (it prints it on failure):

        required_bytes = (m * k * topK * sizeof(int8)) * 3 + 3MB

    where k == hidden. The HCCL default is 200MB, which is too small from
    M=1536 (226MB) upward — the op then fails host-side tiling with
    EZ9999 "HCCL_BUFFSIZE is too SMALL" before any kernel runs. We size
    for the max M in the sweep, add a safety margin, and round up to MB.

    Must run BEFORE HCCL init (setup_all -> init_process_group); HCCL
    reads HCCL_BUFFSIZE once at communicator creation. If the user has
    already exported a larger value we leave it untouched.
    """
    max_m = max(num_tokens_list)
    required_bytes = (max_m * hidden * topk * 1) * 3 + 3 * 1024 * 1024
    # 25% margin, rounded up to whole MB.
    required_mb = ((int(required_bytes * 1.25) + (1 << 20) - 1) >> 20)
    current = os.environ.get("HCCL_BUFFSIZE")
    if current is not None and current.isdigit() and int(current) >= required_mb:
        logger.info("HCCL_BUFFSIZE=%s MB (user-set, >= required %d MB) — keeping",
                    current, required_mb)
        return
    os.environ["HCCL_BUFFSIZE"] = str(required_mb)
    logger.info("HCCL_BUFFSIZE set to %d MB for max M=%d (k=%d topk=%d)",
                required_mb, max_m, hidden, topk)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [r%(rank)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    rank = int(os.environ.get("RANK", "0"))
    logger_filter = logging.Filter()
    logger_filter.filter = lambda r: setattr(r, "rank", rank) or True
    for h in logging.getLogger().handlers:
        h.addFilter(logger_filter)

    logger.info("collector starting: ep_size=%d hidden=%d inter=%d ne=%d topk=%d "
                "tokens=%s quants=%s",
                args.ep_size, args.hidden, args.inter, args.num_experts,
                args.topk, args.num_tokens_list, args.quant_types)

    # Must run before setup_all() -> HCCL init: HCCL_BUFFSIZE is read once
    # at communicator creation.
    _ensure_hccl_buffsize(args.num_tokens_list, args.hidden, args.topk)

    dctx, stack = setup_all(args.ep_size)
    logger.info("dist OK: world=%d ep=%d ep_rank=%d group=%s",
                dctx.world_size, dctx.ep_world_size, dctx.ep_rank, dctx.group_name)

    out_dir = Path(args.output_dir)
    if dctx.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = _csv_path(out_dir, args.ep_size)
        # Default: truncate + rewrite ("w") so a run yields a single-version,
        # duplicate-free file. The old unconditional append ("a") silently
        # stacked every prior smoke test / failed retry onto the same file,
        # mixing measurements from different code versions for the same
        # (M, dtype) key. --append opts back into appending to resume a run.
        mode = "a" if args.append else "w"
        write_header = mode == "w" or not csv_path.exists()
        fh = csv_path.open(mode, newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "op_name", "kernel_source",
                "num_tokens", "hidden", "inter",
                "num_experts", "topk", "ep_size",
                "num_local_experts", "dtype", "latency_us",
            ])
            fh.flush()
        logger.info("writing %s (mode=%s)", csv_path, mode)
    else:
        fh, writer = None, None

    # Order specs by ascending difficulty: M outer (small -> large),
    # quant inner. A device-level error tends to hit only at large M, so
    # this guarantees every small/medium-M point for BOTH quant types is
    # collected and flushed before any crash — instead of the old
    # quant-outer order, where a single large-M bf16 failure buried the
    # entire w8a8 sweep (the one production actually uses).
    specs = [
        DispatchSpec(
            num_tokens=m, hidden=args.hidden, inter=args.inter,
            num_experts=args.num_experts, topk=args.topk,
            ep_world_size=args.ep_size, quant_type=q,
        )
        for m in args.num_tokens_list
        for q in args.quant_types
    ]

    n_ok = 0
    n_fail = 0
    t_start = time.monotonic()
    fatal = False
    try:
        for i, spec in enumerate(specs):
            tag = f"[{i+1}/{len(specs)}] M={spec.num_tokens} {spec.quant_type}"
            meta = None
            try:
                forward_fn, meta = create_dispatch_combine_func(spec, dctx)
                result = benchmark_npu(
                    forward_fn,
                    warmup_iters=args.warmup_iters,
                    num_runs=args.num_runs,
                )
                # Sync across ranks so timing is consistent.
                torch.distributed.barrier(group=dctx.ep_group)
                if dctx.rank == 0:
                    _write_row(writer, spec, result.avg_us)
                    fh.flush()
                    logger.info("%s -> %.2f us (%s)", tag, result.avg_us,
                                "graph" if result.used_graph else "eager")
                n_ok += 1
            except Exception as e:
                logger.exception("%s FAILED: %s", tag, e)
                n_fail += 1
                # A device-level error (HCCL 507057 / SUSPECT REMOTE ERROR /
                # PTA acl api failure) leaves the NPU in an unrecoverable
                # state — every subsequent op, including empty_cache(), will
                # re-raise. Stop the sweep cleanly so the rows already
                # flushed to CSV survive, instead of crashing the process.
                if _is_fatal_device_error(e):
                    logger.error("%s is a device-level error; aborting this "
                                 "ep sweep after %d ok / %d fail (CSV preserved).",
                                 tag, n_ok, n_fail)
                    fatal = True
                    break
            finally:
                # Free per-spec tensors before the next one. Skip on a fatal
                # device error: empty_cache() would itself re-raise 507057.
                if meta is not None:
                    del meta
                if not fatal:
                    try:
                        torch.npu.empty_cache()
                    except Exception:
                        pass
    finally:
        if fh is not None:
            fh.close()
        if not fatal:
            stack.close()

    elapsed = time.monotonic() - t_start
    logger.info("done: %d ok, %d fail, %.1fs%s", n_ok, n_fail, elapsed,
                " (ABORTED on device error)" if fatal else "")


if __name__ == "__main__":
    main()
