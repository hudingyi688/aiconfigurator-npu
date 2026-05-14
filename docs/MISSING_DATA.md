# Missing data on this NPU box

This file records what could **not** be collected on this CANN
8.5.0 / Ascend 910_93 host, and why. AIConfigurator should be aware
of these gaps when consuming the lookup tables produced from
`data/`.

| Op family | BF16 | W8A8 dynamic | Notes |
|---|:---:|:---:|---|
| GEMM (MatMul) | OK (1215 rows) | OK (1215 rows) | Full sweep |
| MoE GroupedMatmul (qwen2/3, deepseek-v2/v2-lite) | OK | OK | Full sweep |
| MoE GroupedMatmul (mixtral-8x7b, mixtral-8x22b) | OK (26 rows) | **Missing (26 specs)** | See "MoE W8A8 mixtral" below |
| ElementWise (RmsNorm/RoPE/SwiGLU/Softmax) | OK (389 rows) | n/a | No quant axis |
| Attention (MLA / SFA / FIA) | **Missing** | **Missing** | See "Attention" below |

---

## Attention

`npu_fused_infer_attention_score` and `npu_sparse_flash_attention`
both fail on this OPP install with:

```
errno[561003] OpName:[SparseFlashAttention_*]
The binary bin not found!
```

`aclnnFusedInferAttentionScoreV3` returns ACL error code 561002 for
**every `head_dim_qk_nope > 128`** combination. A bisection over
`(head_dim, n_heads, sl)` confirmed:

- `head_dim=128`: OK
- `head_dim ∈ {144, 160, 176, 192, 256}`: FAILED(561002)
- `n_heads`, `sl` independent — varying them does not help

CANN 8.5.0 / OPP `Version=8.5.0` (timestamp 20250725) was built with
prebuilt FIA / SFA binaries only for the standard MLA shape
`head_dim=128`. Models that need `head_dim=192` (e.g. GLM-5) cannot
run attention on this box without an OPP upgrade.

**Effect on lookup tables**: no attention rows. AIConfigurator
should fall back to a model-agnostic attention model or use GPU-side
attention numbers when modelling these models on Ascend.

Diagnostic tooling kept for reuse:

- `tools/probe_sfa_standalone.sh` — minimal SFA reproducer
- `tools/probe_sfa_full_error.sh` — capture full ACL error
- `tools/collect_plog_for_sfa.sh` — diff CANN plog around an SFA run
- `AIC_PROBE_FIA=1` — re-enable in-collector FIA dimension probes

## MoE W8A8 mixtral

Triton/BiSheng compilation fails for both mixtral configs with:

```
ub overflow, requires 2621440 bits while 1572864 bits available
Failed to run BiShengHIR pipeline
```

Affected configs (every token size from 1 to 4096 fails):

| Model | hidden | intermediate | experts | topk | Failed |
|---|---:|---:|---:|---:|---:|
| mixtral-8x7b  | 4096 | 14336 | 8 | 2 | 13 / 13 |
| mixtral-8x22b | 6144 | 16384 | 8 | 2 | 13 / 13 |

Cause: Ascend 910_93 SoC has 1.5 MB of Unified Buffer; the
GroupedMatmul W8A8 tiling on `intermediate ≥ 14336` requires
~2.5 MB. Token count does not affect this — even `tokens=1` fails,
i.e. the weight tile alone exceeds UB. All other MoE models in the
sweep have `intermediate ≤ 2816` and pass.

**Effect on lookup tables**: BF16 rows for mixtral are present;
W8A8 rows are missing. AIConfigurator should fall back to a
BF16/W8A8 ratio computed on a similar-shape model (qwen2-moe-57b
or deepseek-v2) when modelling mixtral W8A8 on Ascend.

Diagnostic tooling:

- `tools/inspect_moe_w8a8_errors.sh` — log + checkpoint analysis
- `tools/diff_moe_w8a8_failures.sh` — diff BF16 vs W8A8 specs

---

## Reproduction commands

```sh
# Re-collect non-attention data
bash tools/collect_no_attention.sh
bash tools/collect_w8a8_sweep.sh

# Inspect outputs
bash tools/inspect_no_attention_data.sh

# Re-confirm gaps
bash tools/diff_moe_w8a8_failures.sh
bash tools/collect_plog_for_sfa.sh   # attention
```

## Environment captured at collection time

```
Driver           : 25.2.1 (V100R001C21SPC009B220)
CANN             : 8.5.0  (timestamp 20250725)
OPP              : 8.5.0
Mindstudio       : 8.3.0.B010
SoC              : ascend910_93 (Ascend 910B3 / A3)
ASCEND_OPP_PATH  : /usr/local/Ascend/cann-8.5.0/opp
```
