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
factor[op_kind, ep_size] = profiler_median_latency[op_kind, ep_size]
                         / sol_alpha_beta_latency[op_kind, ep_size]
modeled_latency  = sol_alpha_beta(volume, ep_size, topology)
reported_latency = modeled_latency * factor[op_kind, ep_size]
```

Each EP is **anchored to its own SOL** — we do **not** divide through
ep=1. The earlier version divided through ep=1, which is a degenerate
dispatch path (256 experts on one rank, no real alltoallv traffic) and
produced misleading factors: ep=8 looked 5× faster than the model and
ep=16 looked 2× slower. With per-EP self-anchoring, factors directly
read as "how much slower than alpha-beta this op runs at this EP" —
e.g. 6.67 at ep=16 W8A8 dispatch+combine reflects that crossing the
node boundary on RoCE costs ~6.7× the intra-node SOL prediction.

## Phase split (prefill vs decode)

The alpha-beta SOL model accuracy depends heavily on message size:
prefill messages (M ~ thousands of tokens) are 30-100x larger than
decode messages (M ~ tens of tokens) and the HCCL kernels approach
peak bandwidth in that regime. Mixing prefill profiler data with a
decode SOL reference (or vice versa) produces nonsense factors —
this was the bug in the v2 calibration where W8A8 dispatch+combine
ep=16 came out at 6.67× because it compared a *prefill* profiler
median to a *decode* SOL reference.

Calibration entries are now keyed `{op}@ep{N}@{phase}`. Reference
shape: H=6144, K=8; M=128 for decode, M=4000 for prefill.
`query_comm_calibration(op_kind, ep_size, phase)` looks up
phase-specific factor first, falls back to other-phase same-ep, then
nearest-EP same-phase, then nearest-EP other-phase, finally 1.0.

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

EP coverage: 8, 10, 16. Unmeasured EP sizes fall back to the
factor of the nearest measured EP (ties broken toward the larger EP),
not to 1.0 — silent 1.0 fallback was letting unmeasured EP points look
artificially fast in Pareto search and outranking calibrated points.

## When to refresh

Re-run `tools/build_comm_calibration.py <profiler_root> <output_json>`
when:

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
