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

## Production vs prediction (4k/1.5k benchmark)

Real-host benchmark, 32× Ascend 910B, GLM-5, isl=4096, osl=1536,
production deployment with chunked prefill enabled:

| Concurrency | TPOT_P50 (ms) | QPS  | tokens/s/gpu |
|-------------|---------------|------|--------------|
| 34          | 30.2          | 0.64 | 30.7         |
| 102         | 37.3          | 1.53 | 73.4         |
| 170         | 44.2          | 2.09 | 100.4        |
| 204         | 51.3          | 2.22 | 106.5        |
| 272         | 53.0          | 2.73 | 131.0        |

aic-npu Pareto top-1, same model, total_gpus=32, isl=4000, osl=1000,
ttft=8000, tpot=200:

| Stage                        | Top-1 config           | bs | tokens/s/gpu | gap to prod |
|------------------------------|------------------------|----|--------------|-------------|
| baseline (no calibration)    | tp8 dp4 ep32           | 24 | 27.94        | -75%        |
| v2 (single-phase, wrong M)   | tp8 dp4 ep32           | 20 | 22.80        | -78%        |
| **v3 (phase-split)**         | **tp2 dp16 ep32**      | 12 | **80.65**    | **-25%**    |

Calibration takes the prediction error from 4× too pessimistic to
~1.3× too pessimistic, **and** flips the recommended config from
tp8 dp4 (sub-optimal) to tp2 dp16 (matches the regime production
actually runs in).

The ~25% residual is **systematic and not addressable by calibration**;
see Blind spots below.

## Blind spots

These are gaps the multiplicative-factor approach cannot close. They
explain why predictions stay ~25% below real-host throughput at the
optimum and why predictions diverge further on long-context inputs.

1. **Attention/comm overlap**. vllm-ascend uses CUDA-graph-style
   capture and overlaps comm with the next layer's compute. The
   analytical model sums the two as if serial. No factor on a comm
   op can recover this — overlap is a scheduling property, not a
   per-op latency property.
2. **Chunked prefill**. Production prefill is chunked and interleaved
   with decode. TTFT in the model assumes a single contiguous prefill
   pass, so model TTFT grows linearly with isl while real TTFT grows
   sub-linearly (the 4k/60k production runs differ by ~3.6× TTFT for
   ~15× isl).
3. **TPOT-vs-isl inversion**. Real TPOT decreases slightly as isl
   grows (37 ms → 30 ms from 5k to 60k input) because the scheduler
   amortizes fixed overheads across more tokens. The analytical
   model has TPOT growing with KV cache length (attention SOL ~ s).
   Calibration cannot reverse this trend.
4. **Single-shape profiler reference**. Each `(op, ep, phase)` factor
   is anchored at one (M, H, K) point. HCCL latency is roughly
   alpha + beta·msg_size, so the ratio profiler/SOL drifts with
   actual message size. We accept this and re-anchor only when the
   reference shape moves to a new band (e.g. H=6144 → H=8192).
5. **EP coverage holes**. profiler data covers ep ∈ {8, 10, 16};
   ep=2/4 fall back to ep=8, ep=24/32 fall back to ep=16. Topology
   crossings inside those ranges (e.g. ep=8 fully intra-node vs
   ep=16 cross-node) bias the fallback in either direction.
6. **DSA module data is single-point**. `dsa_generation_module_perf.txt`
   only covers num_heads=64 (TP=1). Search at TP>1 forces HYBRID
   mode for the attention component, so attention latency in the
   prediction is itself SOL+empirical, not silicon. Calibration has
   no signal there to work with.

For accuracy improvements beyond ~25%, build a scheduling-aware
runtime model (overlap, chunked prefill, KV scheduling). That is out
of scope for the perf-database approach.

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
