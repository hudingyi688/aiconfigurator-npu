# GLM-5 Profiler-vs-aic-npu Coverage Analysis

Status: 2026-05-28
Branch: `local/dsa-debug-snapshot`

## TL;DR

aic-npu's silicon coverage of GLM-5's actual production hot path:

| phase  | covered (silicon)            | covered (calibration)    | **uncovered**                |
|--------|------------------------------|--------------------------|------------------------------|
| prefill| 6.6%                         | 4.0%                     | **87.2%** (KV transfer)      |
| decode | 47.4%                        | 46.9%                    | **5.7%** (TP>1 DSA + misc)   |

The prefill 87% gap is a single category — KV transfer between
prefill and decode workers in the PD-disaggregated deployment. It
exists because aic-npu's design premise is "system latency = sum
of operator latencies," and KV transfer isn't an operator. It's a
scheduler + IPC + HCCL behavior that cannot be unit-benchmarked.

This makes the current `disagg` Pareto recommendation directionally
correct (config shape matches production after pinning prefill tp=16
/ decode tp=4) but **systematically pessimistic by ~50%** on absolute
throughput, because half of prefill time-share is unaccounted for.

## How the gap was measured

Source: 11 production profiler runs at
`/Users/hudingyi/Downloads/仿真/glm5-profiler.tar.gz`, taken on a
PD-disaggregated 32-card deployment running GLM-5-w8a8 with chunked
prefill, mooncake KV transfer, and MTP=2.

Each profiler run's `kernel_details.csv` was parsed, kernels grouped
into op families by name pattern, total `Duration(us)` summed per
family, then divided by total run duration to get time-share.

### Prefill (114 s total device time, ep=16)

| op family                       | time-share | aic-npu source           |
|---------------------------------|-----------:|--------------------------|
| **KV_BROADCAST_AICPU**          |  **32.0%** | unmodeled                |
| **KV_REDUCESCATTER_AICPU**      |  **27.8%** | unmodeled                |
| **KV_BROADCAST_HCCL**           |  **27.4%** | unmodeled                |
| GEMM_W8A8                       |       4.3% | silicon                  |
| MOE_DISPATCH_W8A8               |       4.0% | calibration only         |
| HCCL_REDUCESCATTER              |       1.7% | silicon                  |
| DSA_ATTN                        |       0.6% | silicon (TP=1 only)      |
| AIV_OTHER                       |       0.5% | unmodeled                |

Three KV-transfer families dominate prefill — combined 87.2% of
device time. They are produced by the mooncake KV connector
streaming paged-attention blocks from prefill workers to decode
workers, plus HCCL broadcasts coordinating the transfer.

### Decode (108 s total device time, ep=8/10)

| op family                       | time-share | aic-npu source           |
|---------------------------------|-----------:|--------------------------|
| MOE_DISPATCH_W8A8               |  **46.9%** | calibration only         |
| GEMM_W8A8                       |  **46.8%** | silicon                  |
| DSA_ATTN                        |       2.2% | silicon (TP=1 only)      |
| GEMM_BF16                       |       0.6% | silicon                  |

Decode is essentially fully covered modulo the TP>1 DSA gap (the
profiler runs were at TP=4 where num_heads_per_rank=16, but the
silicon DSA module table only has num_heads=64; HYBRID mode patches
this with SOL+empirical).

## Operator-level mapping

Beyond the time-share view, here is which profiler operators map to
which aic-npu modeling source. Confirmed by inspecting `Input Shapes`
in `kernel_details.csv` and matching against the corresponding
`query_*` method in `perf_database.py`.

### Mapped to silicon

| profiler kernel                                | profiler shape (typical)         | aic-npu op                      |
|------------------------------------------------|----------------------------------|---------------------------------|
| `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3`  | (256, 6144) × (4096, 6144)       | `query_gemm` W8A8               |
| `aclnnMatmul_MatMulCommon_MatMulV2`            | (256, 6144) × (256, 6144)        | `query_gemm` BF16               |
| `MatMulV2_ND_ND_FP16_FP16_false_true_all`      | (3, 6144) × (128, 6144)          | `query_gemm` BF16               |
| `SparseFlashAttention`                         | (256, 64, 512) ×  (1681, 128, 1, 512) | `query_*_dsa_module`        |
| `LightningIndexer`                             | (256, 32, 128) × (1681, 128, 1, 128)  | `query_*_dsa_module` (inner)|
| `batch_matmul_transpose`                       | DSA pre/post BMM                 | `query_*_dsa_module` (inner)    |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | (64, 256, 192) × (64, 192, 512)  | `query_mla_bmm`                 |
| `hcom_allReduce_*`                             | small volume                     | `query_custom_allreduce`        |
| `hcom_allGather_*`                             | medium                           | `query_nccl(all_gather)`        |
| `hcom_reduceScatter_*`                         | medium                           | `query_nccl(reduce_scatter)`    |

### Mapped via calibration (no silicon)

| profiler kernel                                | profiler shape                   | aic-npu op                      |
|------------------------------------------------|----------------------------------|---------------------------------|
| `DispatchFFNCombine`                           | (256, 6144) prefill / (1, 6144) decode | `MoEDispatch.query` + `query_comm_calibration("moe_dispatch_combine_w8a8", ep, phase)` |
| `MoeDistributeDispatchV2`                      | (1, 6144) decode                 | calibration `moe_dispatch_bf16` |
| `MoeDistributeCombineV2`                       | (1, 6144) decode                 | calibration `moe_combine_bf16`  |
| `hcom_alltoall_*` / `hcom_alltoallv_*`         | varies                           | calibration `all_to_all` (SOL fallback) |

### Unmapped (= unmodeled)

| profiler kernel                                | time-share (prefill / decode) | nature                        |
|------------------------------------------------|------------------------------|-------------------------------|
| `broadcastAicpuKernel` + `hcom_broadcast_`     | **32% + 27% prefill**         | mooncake KV producer push     |
| `reduce_scatterAicpuKernel`                    | **28% prefill**              | KV pool sync                  |
| `mla_preprocess_0_mix_aic`                     | small (decode)               | fused MLA preproc, separable  |
| `MoeGatingTopK`                                | small                        | router, separable             |
| `ReshapeAndCacheNdKernel` / `ScatterNdUpdate`  | small                        | KV write, separable           |
| `_triton_rope_siso`                            | small                        | RoPE, separable               |
| `SwiGlu` / `AddRmsNormBias`                    | small                        | activation+norm, in elementwise SOL |
| `aclnnDynamicQuantV2` / `AscendQuantV3`        | small                        | quantization util, in SOL     |
| `PadV3` / `MemSet` / `aclnnInplaceCopy`        | small                        | memory ops, untracked         |
| `ApplyTopKTopPCustom`                          | small (decode)               | sampling, untracked           |

The **separable** unmapped ops (≈5% combined time-share) can be
covered by extending the existing `collector/npu/` collectors —
they're individual kernels with stable shapes. The **non-separable**
unmapped ops (KV transfer at 87% of prefill) cannot.

## Why KV transfer cannot be collected as an operator

aic-npu's collector pattern is to import torch + torch_npu +
vllm-ascend kernels, build the input tensors directly, and time
single kernel calls. This works for self-contained ops:
`aclnn_QuantMatmulV3`, `npu_moe_distribute_dispatch`,
`npu_flash_attention_score`, etc. — anywhere the kernel is the
entire unit of work.

KV transfer in the production deployment is *not* a kernel call:

1. The prefill scheduler emits a `kv_producer` event per finished
   layer.
2. The mooncake connector picks up the event, looks up the target
   decode rank's HCCL group, and issues an async `hcom_broadcast`
   on a paged-attention block.
3. The decode worker's `kv_consumer` connector schedules the
   incoming block into its KV pool.
4. HCCL completes the transfer; the AICPU on both sides drives
   `broadcastAicpuKernel` / `reduce_scatterAicpuKernel` for the
   coordination handshake.

The `hcom_broadcast_*` kernel itself *can* be benched in isolation
(it's just an HCCL collective), but its *frequency* and *volume*
in production are determined by the scheduler — how the chunked
prefill is sliced, how often KV blocks are produced, how prefix-
cache hits short-circuit transfers, how the kv_pool's flow control
back-pressures the producer. None of this is captured by per-
kernel timing.

The same limitation applies to:

- **Prefix cache hit rate** — affects how much KV needs to be
  computed and transferred at all.
- **Chunked prefill slicing** — `--max-num-batched-tokens=4096`
  changes the per-step input shape vs. unbounded prefill.
- **MTP draft acceptance rate** — affects effective decode steps
  per generated token.
- **Mooncake KV pool lock contention** — varies with concurrency
  and request size distribution.

These are scheduler-level properties of vllm + the deployment
configuration. To capture them you need to either (a) run the real
service and profile end-to-end, or (b) build a scheduler-aware
runtime model on top of the per-op latencies.

## Comparison to msmodeling-style profiling

This is a structural difference between the two methodologies, not
a deficiency of either:

| dimension                    | aic-npu (collector-based)    | msmodeling (service-profile-based) |
|------------------------------|------------------------------|-------------------------------------|
| data collection              | direct kernel calls, no service | full vllm/sglang service          |
| operator granularity         | precise (M, N, K, dtype)     | usually layer-aggregated            |
| scheduler behaviors          | invisible                    | natively included                   |
| collection cost              | low (minutes per shape band) | high (full deployment + warmup)     |
| data stability               | high (deterministic kernel)  | medium (scheduling jitter)          |
| cross-config interpolation   | analytical model × op db     | requires re-profiling per config    |

aic-npu's strength: cheap, fine-grained data, with analytical models
that interpolate cleanly across (M, N, K, ep, dp, batch) sweeps.
Production deployment shape decisions land here.

aic-npu's weakness: blind to anything the scheduler does. The
production-aligned absolute-throughput numbers land in msmodeling
or end-to-end benchmarks.

The two approaches are **complementary**, not competing. A useful
production stack uses aic-npu as the configuration-space search
engine ("what shape should we deploy") and a service-level profiler
as the calibration ground truth ("how fast does that shape actually
run").

## What this means for the GLM-5 deployment recommendation

Given the gap analysis, here's what the current aic-npu output is
useful for and not useful for:

**Trustworthy:**
- **Configuration shape** (tp, dp, ep, agg vs disagg). Once prefill
  tp=16 and decode tp=4 are pinned (commit `6849445`), the search
  reproduces the production deployment shape. Sweeping `bs / dp / ep`
  around that pinned shape finds bigger-batch / different-dp variants
  with comparable trustworthiness.
- **Relative ranking** between configs at fixed deployment family.
  If the model says A > B by 30% on tokens/s/gpu, that ranking is
  informative even if the absolute numbers are off.
- **Decode path absolute latency** (within ~25% of production).
  Decode is 94% silicon-or-calibration covered.

**Not trustworthy as-is:**
- **Absolute disagg throughput.** Predictions are systematically
  ~50% low because prefill's KV-transfer 87% is missing. Use a
  multiplicative correction (~1.5-2×) when sizing capacity.
- **Long-context predictions** (isl > 8k). Hits both the KV-transfer
  gap and the unmodeled chunked-prefill scheduling.
- **Comparing PD-disagg to single-node agg.** The model's `agg`
  branch has no KV transfer to remove, so it looks artificially
  closer to ground truth than `disagg` does. The comparison is
  apples-to-oranges.

## Concrete next steps to close the gap (priority order)

### 1. Extend collectors to cover the separable unmapped ops (low cost)

| new collector                       | profiler op                      | est. time-share gained |
|-------------------------------------|----------------------------------|-----------------------|
| `collect_mla_preprocess.py`         | `mla_preprocess_0_mix_aic`       | ~1% decode            |
| `collect_moe_dispatch_bf16.py`      | `MoeDistributeDispatchV2/CombineV2` | replaces calibration → silicon for BF16 path |
| `collect_router.py`                 | `MoeGatingTopK`                  | <1%                   |
| `collect_kv_write.py`               | `ReshapeAndCacheNdKernel`        | <1%                   |
| `collect_rope.py`                   | `_triton_rope_siso`              | <1%                   |

These are all single-kernel ops with deterministic shapes; each
collector is ~half a day patterned on `collect_attn.py`.

### 2. DSA TP>1 silicon (low cost, eliminates one HYBRID dependency)

`mla_module_factory.py` accepts a `--num-heads-override` flag,
re-run for TP ∈ {2, 4, 8, 16} → num_heads ∈ {32, 16, 8, 4},
append to `dsa_*_module_perf.txt`. Removes the SILICON-mode
crash on tp>1 and unblocks pure-SILICON Pareto runs.

### 3. KV transfer model (high impact, requires choices)

Three options, in order of fidelity:

#### 3a. Pure analytical (cheap, weak)

```
KVTransfer_per_request(isl) =
    (num_layers × kv_lora_rank × isl × bytes_per_element) / mooncake_bw
    + alpha_per_request
```

GLM-5 W8A8: `78 × 512 × isl × 1B = 40 KB × isl bytes`. At
mooncake_bw ≈ 25 GB/s and alpha ≈ 1 ms, isl=4000 → 7.4 ms.
**Production observed: ~330 ms/request.** The analytical model is
off by 40-50×, indicating the production traffic pattern is per-
layer, per-step, with broadcast fan-out — not a single bulk transfer.

#### 3b. Mooncake bench (medium, requires NPU)

Run mooncake's own `mooncake-bench` tool standalone (no vllm
needed) to extract real bandwidth and per-call alpha. Plug into a
parametric model with the production-observed access pattern
(per-layer × per-step). Likely lands within 2× of production.

#### 3c. Differential profiler (most accurate, requires one extra collection)

Run a single agg-mode (no PD split, no mooncake) profiler under
the same bench load, take per-shape time differences against the
existing PD profiler. The delta is a direct measurement of mooncake
KV transfer cost as a function of `(isl, ep, batch)`. Fit a
polynomial, ship as a new `KVTransfer.query` op in the disagg
pareto path.

### 4. Long-term: scheduler-aware runtime model

Out of scope for the perf-database approach. Would require building
a discrete-event simulator for vllm's scheduler — paged attention
allocation, chunked prefill slicing, prefix-cache eviction, MTP
draft pipeline, mooncake KV flow control. This is a separate
engineering effort the size of aic-npu itself.

## Summary

The production hot-path coverage gap is not in operator-level data
quality; it's in operator-level scope. aic-npu was designed for a
"compute = sum of kernels" world, and the GLM-5 PD-disaggregated
deployment falls outside that world by spending 87% of prefill in
scheduler-driven KV transfer. Fix path: extend collectors to fully
cover the kernel-shaped portion (priorities 1 + 2), build a
parametric KV-transfer model for the rest (priority 3), accept that
absolute disagg throughput will need ~1.5× correction until that
work lands.
