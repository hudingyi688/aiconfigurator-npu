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
actually runs in). The ~25% residual is **systematic and not
addressable by calibration**; see Blind spots below.

## Long-context behavior

Same setup, isl swept across the production buckets:

| isl (model) | top-1 config       | bs | TPOT (model) | TPOT (prod) | TTFT (model) | TTFT (prod) | tokens/s/gpu (model) | tokens/s/gpu (prod) |
|-------------|--------------------|----|--------------|-------------|--------------|-------------|----------------------|---------------------|
| 5000        | tp4 dp8 ep32       | 16 | 65.8 ms      | 36.8 ms     | 4677 ms      | 1695 ms     | 56.4                 | 73.4                |
| 5000 #2     | tp2 dp16 ep32      | 7  | 55.0 ms      | —           | 6963 ms      | —           | 54.4                 | —                   |
| 15000       | tp8 dp4 ep32       | 2  | 24.8 ms      | 32.6 ms     | 7681 ms      | 2691 ms     | 8.7                  | ~42                 |
| 30000       | OOM in model       | —  | —            | 31.8 ms     | —            | 3756 ms     | —                    | (sim)               |
| 60000       | OOM in model       | —  | —            | 30.2 ms     | —            | 6128 ms     | —                    | (sim)               |

Reading this:

- **TTFT consistently 2.5-3× too pessimistic** (model 4677/7681 vs
  prod 1695/2691). This is Blind spot #2 — production runs chunked
  prefill, the model treats prefill as one contiguous pass.
- **TPOT inverts at ~10k isl**. At 5k the model is pessimistic
  (66 vs 37 ms); at 15k it's *optimistic* (25 vs 33 ms). Different
  failure modes: at 5k, calibrated comm + analytical attention sum
  serially with no overlap (Blind spot #1); at 15k, the model can
  only fit bs=2 on 32 cards while production runs at concurrency=102
  via paged KV — TPOT for bs=2 is just genuinely lower than for a
  saturated batch (Blind spot #6, KV scheduling).
- **OOM at 30k+ isl**. The model accounts for "weights + KV at full
  isl + activations" against HBM. It doesn't model paged-attention
  / KV swap-out / prefix cache. Production runs 60k isl on 32 cards
  routinely; the model rejects it. Long-context Pareto search on
  this stack is **not trustworthy past the OOM threshold**.

**Conclusion on calibration scope**: only the short-context (≤4-5k)
branch of this benchmark stresses the comm-calibration approach. The
~25% throughput residual at 4k is the ceiling of what this method
can deliver. Long-context divergence is dominated by KV scheduling
and chunked prefill, neither of which a per-op multiplicative factor
can capture. Future improvements there require a scheduler-aware
runtime model.

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
   pass, so model TTFT runs ~3× higher than production (4k/1.5k:
   model 4677 vs prod 1695; 15k: model 7681 vs prod 2691). The
   factor approach has no hook to reach this.
3. **TPOT-vs-isl inversion**. Real TPOT decreases slightly as isl
   grows (37 ms at 5k → 30 ms at 60k) because the scheduler
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
6. **No KV scheduling model**. The OOM check is "weights + KV at
   full isl + activations" vs HBM. Production paged attention,
   prefix-cache reuse, and KV swap let real deployments fit isl
   well past the model's threshold (60k on 32 cards is routine in
   prod; the model rejects 30k). Past that threshold predictions
   are not just inaccurate, they are missing — there's no result
   row to compare against. Calibration cannot shift this boundary.
7. **DSA module data is single-point**. `dsa_generation_module_perf.txt`
   only covers num_heads=64 (TP=1). Search at TP>1 forces HYBRID
   mode for the attention component, so attention latency in the
   prediction is itself SOL+empirical, not silicon. Calibration has
   no signal there to work with.
8. **SOL alpha-beta has no alpha term for small messages**.
   The query_nccl SOL formula is purely beta-driven
   (`memory * msg * (n-1)/n / bw`), with no per-call kernel-launch
   constant. For dispatch+combine in decode the actual per-rank input
   is M=1 (verified from kernel_details.csv), giving SOL ~1-3 us
   while profiler reads 100-300 us — a 50-200× ratio that calibration
   *can* express but only by inflating factors to a regime where any
   nearest-EP fallback error is amplified to the same magnitude.
   The current v3 calibration sidesteps this by anchoring SOL at
   M=4000 (prefill) / M=128 (decode) — message sizes large enough
   that beta dominates and factors stay in the 0.1-7 range. This is
   technically inconsistent with the actual profiler shapes but
   keeps the search numerically well-behaved. A correct fix is at
   the SOL model layer (add an alpha term to query_nccl SOL), not
   at the calibration layer.

For accuracy improvements beyond ~25%, build a scheduling-aware
runtime model (overlap, chunked prefill, paged-attention KV,
prefix-cache reuse). That is out of scope for the perf-database
approach.

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
