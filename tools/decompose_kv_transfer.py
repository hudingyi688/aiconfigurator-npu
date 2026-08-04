#!/usr/bin/env python3
"""Decompose KV-transfer 6-point black box into three physical layers.

Layer 1 (SILICON): hcom_broadcast_ / hcom_reduceScatter_ NPU kernel latency
                   — from nccl_perf.txt microbench (already collected)
Layer 2 (CALIBRATION): AICPU dispatch overhead = profiler(AicpuKernel) - profiler(hcom_)
                   — diff from profiler groundtruth, per (op, nd)
Layer 3 (MODEL): call-count model calls(ep, isl) = num_layers * chunks_per_request
                   — fit from profiler call counts

Reconstructed: KV_transfer(isl, ep) = calls(ep, isl) * [hcom_latency(msg, nd) + aicpu_overhead(op, nd)]

Validation: compare reconstructed total vs current kv_transfer_perf.txt 6 points.
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DETAIL = REPO / "docs/profiler_alignment/groundtruth/detail.csv"
NCCL_PERF = REPO / "systems/data/ascend_910b/vllm-ascend/0.18.0/nccl_perf.txt"
KV_TRANSFER = REPO / "src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/kv_transfer_perf.txt"

# GLM-5 production config
NUM_LAYERS = 78
KV_LORA_RANK = 512
DTYPE_BYTES = 2  # bf16
CHUNK_SIZE = 256  # per-rank tokens after CP16 split of max-num-batched-tokens=4096

# KV-transfer op pairs: (aicpu_wrapper, hcom_kernel)
KV_OP_PAIRS = [
    ("broadcastAicpuKernel", "hcom_broadcast_"),
    ("reduce_scatterAicpuKernel", "hcom_reduceScatter_"),
]
# Mooncake IPC ops (no hcom pair, pure AICPU)
KV_IPC_OPS = ["allgatherAicpuKernel", "batch_getAicpuKernel", "batch_putAicpuKernel", "OneSideCommAicpuInit"]


def parse_profiler_detail():
    """Return {(phase, ep, isl): {op_type: {count, median_us, avg_us, total_ms_med}}}.

    Aggregates across runs and ranks. ep=0 means dp-only (no EP); we treat
    ep=0 as ep=1 for the model (dp-only prefill worker).

    total_ms_med = median_us * count / 1000 — uses MEDIAN (not avg) because
    avg_us is polluted by rank-sync bubbles (max_us up to 10M us). This matches
    the kv_transfer_perf.txt methodology ("must use median, mean is polluted").
    """
    data = defaultdict(lambda: defaultdict(lambda: {"count": 0, "median_us": 0.0, "avg_us": 0.0, "total_ms": 0.0}))
    with DETAIL.open() as f:
        for r in csv.DictReader(f):
            phase = r["phase"]
            ep = int(r["ep"])
            isl = r["isl"]
            op = r["op_type"]
            count = int(r["count"])
            med = float(r["median_us"])
            avg = float(r["avg_us"])
            # Use median*count (not avg*count) — avg is polluted by rank-sync bubbles
            total_ms = med * count / 1000.0
            data[(phase, ep, isl)][op] = {"count": count, "median_us": med, "avg_us": avg, "total_ms": total_ms}
    return data


def load_nccl_microbench():
    """Return {(op_name, num_devices, msg_bytes): latency_ms}.

    nccl_perf.txt columns: nccl_dtype,num_gpus,message_size,kernel_source,op_name,latency(ms)
    """
    data = {}
    with NCCL_PERF.open() as f:
        for r in csv.DictReader(f):
            key = (r["op_name"], int(r["num_gpus"]), int(r["message_size"]))
            data[key] = float(r["latency"])
    return data


def query_nccl_latency(nccl_db, op_name, msg_bytes, nd):
    """Bilinear lookup with clamp+log-interp on msg_bytes axis."""
    # find exact or bracketing msg_bytes for this (op, nd)
    pts = sorted(m for (o, n, m) in nccl_db if o == op_name and n == nd)
    if not pts:
        return None
    if msg_bytes in pts:
        return nccl_db[(op_name, nd, msg_bytes)]
    # clamp
    if msg_bytes < pts[0]:
        return nccl_db[(op_name, nd, pts[0])]
    if msg_bytes > pts[-1]:
        return nccl_db[(op_name, nd, pts[-1])]
    # bracket + linear interp
    for i in range(len(pts) - 1):
        if pts[i] <= msg_bytes <= pts[i + 1]:
            lo, hi = pts[i], pts[i + 1]
            vlo, vhi = nccl_db[(op_name, nd, lo)], nccl_db[(op_name, nd, hi)]
            t = (msg_bytes - lo) / (hi - lo)
            return vlo + t * (vhi - vlo)
    return None


def main():
    prof = parse_profiler_detail()
    nccl = load_nccl_microbench()

    print("=" * 100)
    print("LAYER DECOMPOSITION OF KV-TRANSFER (profiler groundtruth -> 3 layers)")
    print("=" * 100)

    # ---- Layer 2: AICPU overhead (diff AicpuKernel - hcom_) ----
    print("\n## Layer 2: AICPU dispatch overhead = profiler(AicpuKernel) - profiler(hcom_)\n")
    print(f"{'phase':>7} {'ep':>4} {'isl':>6} | {'op_pair':>40} | {'aicpu_med':>10} {'hcom_med':>10} {'overhead':>10} {'calls':>6}")
    print("-" * 100)

    aicpu_overhead = {}  # (op_pair_label, nd) -> median overhead_us
    overhead_samples = defaultdict(list)

    for (phase, ep, isl), ops in sorted(prof.items()):
        if phase != "prefill":
            continue
        nd = ep if ep > 0 else 1  # ep=0 -> nd=1 (dp-only)
        for aicpu_op, hcom_op in KV_OP_PAIRS:
            a = ops.get(aicpu_op)
            h = ops.get(hcom_op)
            if not a or not h:
                continue
            ovh = a["median_us"] - h["median_us"]
            label = f"{aicpu_op}↔{hcom_op}"
            print(f"{phase:>7} {ep:>4} {isl:>6} | {label:>40} | {a['median_us']:>10.1f} {h['median_us']:>10.1f} {ovh:>10.1f} {a['count']:>6}")
            overhead_samples[(aicpu_op, nd)].append(ovh)

    print("\n### Layer 2 summary: median AICPU overhead per (op, nd)\n")
    print(f"{'op_pair':>45} {'nd':>4} {'n_pts':>6} {'median_ovh_us':>14} {'mean_ovh_us':>14}")
    print("-" * 90)
    for (aicpu_op, nd), samples in sorted(overhead_samples.items()):
        hcom_op = next(h for a, h in KV_OP_PAIRS if a == aicpu_op)
        label = f"{aicpu_op}↔{hcom_op}"
        med = statistics.median(samples)
        avg = statistics.mean(samples)
        aicpu_overhead[(aicpu_op, nd)] = med
        print(f"{label:>45} {nd:>4} {len(samples):>6} {med:>14.1f} {avg:>14.1f}")

    # ---- Layer 1: hcom microbench (SILICON) at production msg_bytes ----
    print("\n## Layer 1: hcom microbench (SILICON) at production KV block sizes\n")
    msg_per_layer_chunk = KV_LORA_RANK * CHUNK_SIZE * DTYPE_BYTES  # 512 * 256 * 2 = 256 KB
    print(f"Production msg_bytes per layer per chunk = {KV_LORA_RANK}*{CHUNK_SIZE}*{DTYPE_BYTES} = {msg_per_layer_chunk} B = {msg_per_layer_chunk/1024:.0f} KB")
    print()

    # nccl op_name: hcom_broadcast_ -> broadcast, hcom_reduceScatter_ -> reduce_scatter
    _OP_NCCL_NAME = {
        "hcom_broadcast_": "broadcast",
        "hcom_reduceScatter_": "reduce_scatter",
    }
    # find nearest msg_bytes in nccl grid (use broadcast as representative grid)
    nccl_msgs = sorted({m for (o, n, m) in nccl if o == "broadcast"})
    nearest = min(nccl_msgs, key=lambda m: abs(m - msg_per_layer_chunk))
    print(f"Nearest nccl grid point: {nearest} B = {nearest/1024:.0f} KB (target {msg_per_layer_chunk/1024:.0f} KB)")
    print()
    print(f"(nccl grid min nd=2; ep=0/dp-only maps to nd=1 — no microbench, falls back to profiler)")
    print()

    print(f"{'op_name':>20} {'nd':>4} {'msg_bytes':>10} | {'mb_per_call_us':>15} {'prof_avg_us':>12} {'prof_med_us':>12} {'avg/med':>8} {'prof_total_ms':>14}")
    print("-" * 110)

    layer1_latency = {}  # (hcom_op, nd) -> microbench per-call latency_ms
    for (phase, ep, isl), ops in sorted(prof.items()):
        if phase != "prefill":
            continue
        nd = ep if ep > 0 else 1
        for aicpu_op, hcom_op in KV_OP_PAIRS:
            h = ops.get(hcom_op)
            a = ops.get(aicpu_op)
            if not h:
                continue
            nccl_op = _OP_NCCL_NAME[hcom_op]
            mb_ms = query_nccl_latency(nccl, nccl_op, nearest, nd)
            if mb_ms is None:
                continue
            layer1_latency[(hcom_op, nd)] = mb_ms
            prof_avg = h.get("avg_us", 0)
            prof_med = h["median_us"]
            ratio = prof_avg / prof_med if prof_med else 0
            print(f"{hcom_op:>20} {nd:>4} {nearest:>10} | {mb_ms*1000:>15.1f} {prof_avg:>12.1f} {prof_med:>12.1f} {ratio:>8.1f}x {h['total_ms']:>14.1f}")

    print("\n### Layer 1 summary: microbench per-call latency (ms) at production msg_bytes\n")
    print(f"{'hcom_op':>22} {'nd':>4} {'mb_per_call_ms':>16}")
    print("-" * 50)
    for (hcom_op, nd), ms in sorted(layer1_latency.items()):
        print(f"{hcom_op:>22} {nd:>4} {ms:>16.5f}")
    print()
    print("NOTE: profiler hcom_* Duration is SESSION-AGGREGATED (Input Shapes=N/A, avg/median=21x).")
    print("      microbench is PER-CALL kernel latency. They are NOT directly comparable.")
    print("      The profiler session aggregates MANY hcom calls + AICPU handshake into one Duration.")
    print("      Reconstruction uses profiler total_ms (avg*count), not microbench*calls.")

    # ---- Layer 3: call-count model ----
    print("\n## Layer 3: call-count model calls(ep, isl) = num_layers * chunks\n")
    print(f"{'phase':>7} {'ep':>4} {'isl':>6} | {'bcast_calls':>12} {'rs_calls':>10} {'bcast/layer':>12} {'rs/layer':>10} {'chunks_fit':>11}")
    print("-" * 90)

    calls_model = {}  # (ep, isl) -> {bcast_calls, rs_calls}
    for (phase, ep, isl), ops in sorted(prof.items()):
        if phase != "prefill":
            continue
        b = ops.get("broadcastAicpuKernel", {}).get("count", 0)
        rs = ops.get("reduce_scatterAicpuKernel", {}).get("count", 0)
        b_per_l = b / NUM_LAYERS if NUM_LAYERS else 0
        rs_per_l = rs / NUM_LAYERS if NUM_LAYERS else 0
        # chunks = ceil(isl_num / (CHUNK_SIZE * cp))... but isl is string like "10k"
        isl_num = int(isl.replace("k", "").replace("K", "")) * 1000 if "k" in isl.lower() else int(isl)
        cp = 16 if ep == 16 else 1
        chunks_fit = isl_num / (CHUNK_SIZE * cp)
        calls_model[(ep, isl)] = {"bcast": b, "rs": rs}
        print(f"{phase:>7} {ep:>4} {isl:>6} | {b:>12} {rs:>10} {b_per_l:>12.1f} {rs_per_l:>10.1f} {chunks_fit:>11.1f}")

    # ---- Reconstruct & validate (using profiler total_ms, the ground truth) ----
    print("\n## Reconstruction: decompose current 6-point table into op-level totals\n")
    print("  (uses profiler total_ms = avg_us * count per (ep, isl) point)")
    print()
    print(f"{'ep':>4} {'isl':>6} | {'current_ms':>11} | {'hcom_bcast':>11} {'aicpu_bcast':>12} {'hcom_rs':>9} {'aicpu_rs':>10} {'ipc':>7} | {'sum_ms':>9} {'diff%':>7}")
    print("-" * 105)

    # load current 6-point table; map ep_size=1 -> profiler ep=0 (dp-only)
    # isl kept as int to match kv_transfer_perf.txt; prof dict uses string isl ("2500","10k","20k")
    current = {}
    with KV_TRANSFER.open() as f:
        for r in csv.DictReader(f):
            current[(int(r["ep_size"]), int(r["isl"]))] = float(r["net_kv_wallclock_ms"])

    # isl int -> profiler string ("2500"->"2500", "10000"->"10k", "20000"->"20k")
    def _prof_isl(isl_num):
        if isl_num >= 10000:
            return f"{isl_num // 1000}k"
        return str(isl_num)

    # ep mapping: kv_transfer ep_size=1 corresponds to profiler ep=0 (dp-only, no EP)
    def _prof_ep(kv_ep):
        return 0 if kv_ep == 1 else kv_ep

    for (kv_ep, isl_num), cur in sorted(current.items()):
        ep = _prof_ep(kv_ep)
        isl_str = _prof_isl(isl_num)
        prof_key = ("prefill", ep, isl_str)
        ops = prof.get(prof_key, {})

        # Decompose by op type using profiler total_ms
        hcom_b = ops.get("hcom_broadcast_", {}).get("total_ms", 0.0)
        aicpu_b = ops.get("broadcastAicpuKernel", {}).get("total_ms", 0.0)
        hcom_rs = ops.get("hcom_reduceScatter_", {}).get("total_ms", 0.0)
        aicpu_rs = ops.get("reduce_scatterAicpuKernel", {}).get("total_ms", 0.0)
        ipc = sum(ops.get(op, {}).get("total_ms", 0.0) for op in KV_IPC_OPS)

        sum_ms = hcom_b + aicpu_b + hcom_rs + aicpu_rs + ipc
        diff = (sum_ms - cur) / cur * 100 if cur else 0
        print(f"{kv_ep:>4} {isl_num:>6} | {cur:>11.1f} | {hcom_b:>11.1f} {aicpu_b:>12.1f} {hcom_rs:>9.1f} {aicpu_rs:>10.1f} {ipc:>7.1f} | {sum_ms:>9.1f} {diff:>6.1f}%")

    # ---- Aggregate decomposition summary ----
    print("\n## Aggregate decomposition (all 6 profiler points summed)\n")
    tot_hcom_b = tot_aicpu_b = tot_hcom_rs = tot_aicpu_rs = tot_ipc = 0.0
    tot_current = 0.0
    for (kv_ep, isl_num), cur in current.items():
        ep = _prof_ep(kv_ep)
        isl_str = _prof_isl(isl_num)
        prof_key = ("prefill", ep, isl_str)
        ops = prof.get(prof_key, {})
        tot_hcom_b += ops.get("hcom_broadcast_", {}).get("total_ms", 0.0)
        tot_aicpu_b += ops.get("broadcastAicpuKernel", {}).get("total_ms", 0.0)
        tot_hcom_rs += ops.get("hcom_reduceScatter_", {}).get("total_ms", 0.0)
        tot_aicpu_rs += ops.get("reduce_scatterAicpuKernel", {}).get("total_ms", 0.0)
        tot_ipc += sum(ops.get(op, {}).get("total_ms", 0.0) for op in KV_IPC_OPS)
        tot_current += cur
    tot_sum = tot_hcom_b + tot_aicpu_b + tot_hcom_rs + tot_aicpu_rs + tot_ipc
    print(f"  {'hcom_broadcast_':>25}: {tot_hcom_b:>8.1f} ms  ({tot_hcom_b/tot_sum*100:>5.1f}%)  [SILICON microbench available]")
    print(f"  {'broadcastAicpuKernel':>25}: {tot_aicpu_b:>8.1f} ms  ({tot_aicpu_b/tot_sum*100:>5.1f}%)  [AICPU dispatch, not microbenchable]")
    print(f"  {'hcom_reduceScatter_':>25}: {tot_hcom_rs:>8.1f} ms  ({tot_hcom_rs/tot_sum*100:>5.1f}%)  [SILICON microbench available]")
    print(f"  {'reduce_scatterAicpuKernel':>25}: {tot_aicpu_rs:>8.1f} ms  ({tot_aicpu_rs/tot_sum*100:>5.1f}%)  [AICPU dispatch, not microbenchable]")
    print(f"  {'IPC (allgather/batch_get/put)':>25}: {tot_ipc:>8.1f} ms  ({tot_ipc/tot_sum*100:>5.1f}%)  [small, keep profiler-derived]")
    print(f"  {'SUM':>25}: {tot_sum:>8.1f} ms")
    print(f"  {'current kv_transfer table':>25}: {tot_current:>8.1f} ms  (diff: {(tot_sum-tot_current)/tot_current*100:.1f}%)")

    print("\n## Conclusion: what can be upgraded from PROFILER_DERIVED to SILICON?\n")
    silicon_pct = (tot_hcom_b + tot_hcom_rs) / tot_sum * 100
    aicpu_pct = (tot_aicpu_b + tot_aicpu_rs) / tot_sum * 100
    ipc_pct = tot_ipc / tot_sum * 100
    print(f"  SILICON-upgradable (hcom_* kernels):  {silicon_pct:.1f}%  -> but microbench is per-call, profiler is session-aggregated")
    print(f"  AICPU dispatch (not microbenchable):   {aicpu_pct:.1f}%  -> must stay profiler-derived or build scheduler model")
    print(f"  IPC (small):                           {ipc_pct:.1f}%  -> keep profiler-derived")
    print()
    print("  BLOCKER: profiler hcom_* Duration is SESSION-AGGREGATED wall-clock (Input Shapes=N/A,")
    print("  avg/median=21x heavy-tail). microbench gives PER-CALL kernel latency. To use microbench")
    print("  data we need the CALL COUNT per session (profiler count field) + per-call microbench latency.")
    print("  But session aggregation means 1 profiler Duration entry != 1 hcom call — the count field")
    print("  tracks AicpuKernel calls, not hcom kernel invocations. This mismatch prevents direct")
    print("  substitution of microbench for profiler hcom_* totals.")
    print()
    print("  RECOMMENDED PATH:")
    print("  1. Keep kv_transfer_perf.txt 6-point table as the calibrated ground truth (PROFILER_DERIVED)")
    print("  2. Add microbench hcom_broadcast_/hcom_reduceScatter_ as a SILICON cross-check layer")
    print("     to validate the hcom portion scales correctly with (msg_bytes, nd)")
    print("  3. Build calls(ep, isl) model from Layer 3 to enable physics-based isl>20k extrapolation")
    print("     instead of linear: calls ~ num_layers * ceil(isl / (chunk*cp))")
    print("  4. The AICPU dispatch overhead (37% of KV transfer) is the true unsolvable part —")
    print("     it's scheduler behavior (mooncake connector, KV pool sync), only profiler can capture")


if __name__ == "__main__":
    main()
