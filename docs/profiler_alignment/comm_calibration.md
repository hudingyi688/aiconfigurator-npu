# MoE Comm Calibration — vllm-ascend

## Why this exists

aiconfigurator's `MoEDispatch` op normally derives latency from an
analytical alpha-beta model fed with measured `nccl_perf.txt` /
`custom_allreduce_perf.txt`. On vllm-ascend two things break that:

1. **No silicon `all_to_all` data**: the HCCL bench we ship doesn't
   produce `all_to_all` rows, so `query_nccl(..., "all_to_all")` only
   has a SOL fallback — pure topology arithmetic, no measured constant.
2. **`dispatch_ffn_combine` is unbenchable in isolation**: the kernel
   hardcodes `max_output_size=65536` and allocates per-rank scratch
   sized to that worst case. Synthetic single-process benchmarks OOM
   even at M=8 on world=1, world=2 and world=16.

The dispatch+combine kernel is the dominant comm cost on the GLM-5
generation path, so leaving it modeled by SOL alone (or by the wrong
collective) makes Pareto search deviate by 15-30% from production.

## Approach

We do **not** copy profiler latencies into `*_perf.txt`. Production
shape, batch composition, and topology move; baking observed latency
into the perf database would freeze a snapshot.

Instead we keep the analytical model and apply a per-`(op_kind, ep_size)`
multiplicative factor on top:

```
modeled_latency  = alpha_beta(volume, ep_size, topology)
reported_latency = modeled_latency * factor[op_kind, ep_size]
```

The factor is the ratio of profiler-median latency at that EP size to
the profiler-median at a baseline EP size for the same op. Because the
ratio is normalized by the model itself at the baseline point, the
factor stays meaningful when message sizes change — it captures the
topology delta (intra-node HCCS vs cross-node RoCE) that the
analytical model misses, not the message-size scaling that the model
already gets right.

## Where the data lives

- File: `src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/comm_calibration.json`
- Loaded by: `PerfDatabase.__init__` (best-effort; absence is silently
  fine and reverts to factor=1.0 for everything).
- Queried by: `PerfDatabase.query_comm_calibration(op_kind, ep_size)`.
- Applied in: `operations.MoEDispatch.query()` vllm-ascend branch.

Op kinds currently calibrated:

| op_kind                       | What it scales                              |
|-------------------------------|---------------------------------------------|
| `moe_dispatch_combine_w8a8`   | fused dispatch+FFN+combine (W8A8 path)      |
| `moe_dispatch_bf16`           | unfused dispatch (BF16 path, pre_dispatch)  |
| `moe_combine_bf16`            | unfused combine (BF16 path, post_dispatch)  |
| `all_reduce`                  | TP-1 attention all-reduce                   |
| `all_gather`                  | DP-1 attention all-gather                   |
| `reduce_scatter`              | DP-1 attention reduce-scatter               |
| `all_to_all`                  | (reserved; not yet wired through MoEDispatch)|

EP coverage: 1, 8, 10, 16. Out-of-range EP returns 1.0 (no correction).

## When to refresh

Re-run the profiler aggregation (`tools/build_comm_calibration.py` —
TODO; the current JSON was hand-written from the
`/tmp/glm5_profiler/glm-profiler/` runs of 2026-05-28) when:

- HCCL / CANN / vllm-ascend / driver versions change,
- EP topology changes (more nodes, different intra-node interconnect),
- model shape changes enough to land in a different message-size band
  (e.g. moving from H=6144 to H=8192).

Stale calibration is detectable: align profiler ground-truth against
the perf-db prediction (`tools/check_alignment.py`) and refresh when
the residual on `MoEDispatch` exceeds ~10%.

## Limitations

- BF16 dispatch+combine is not yet integrated into the collector.
  We rely on profiler-derived calibration end-to-end for that path.
- EP=2 and EP=4 currently fall back to factor=1.0; once we have
  profiler data on a 2-card and 4-card slice, fill those entries.
- `all_to_all` is reserved as a separate kind even though it
  currently lands at the same SOL call as the W8A8 path; future
  schedulers may use a non-fused all-to-all where this matters.
