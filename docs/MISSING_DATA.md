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
| Attention (MLA / SFA / FIA) | OK | TBD | DSA SFA path unblocked — see "Attention" below |

---

## Attention

> **Status (2026-05-26)**: previous "no attention rows" diagnosis was
> wrong. The DSA SFA path runs end-to-end after fixing the OOT custom-op
> dispatch order. See "Resolved root cause" below. Full sweep is
> running; this section is kept for historical reference.

### Resolved root cause

The collector raised
`errno 561003 OpName:[SparseFlashAttention_*] The binary bin not found`
and (after the OOT path was wired up) a kernel segfault with
`runtime_attrs.cc: Failed to get attr, the index 5/6/7/8 out of range 5`.

vllm-ascend ships its **own** SparseFlashAttention op:

```
csrc/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp
  Attr count = 5  (scale_value, sparse_block_size, layout_query,
                   layout_kv, sparse_mode)
```

Its prebuilt OOT bin lives under

```
site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/
  op_impl/.../sparse_flash_attention/SparseFlashAttention_*.json
  (5 attrs, matches the op def)
```

CANN 8.5.0 also ships a SparseFlashAttention bin — with **9 attrs** —
under `opp/built-in/.../sparse_flash_attention/`. ACL runtime picks
whichever vendor matches the dispatch key first. If
`ASCEND_CUSTOM_OPP_PATH` is unset (or set after `import torch_npu`,
which happens in the collector's startup hook because
`NPUPlatform.import_kernels()` runs late), ACL dispatches the
5-attr call into the 9-attr CANN binary, the kernel reads attrs
5/6/7/8 past the op def, and the process segfaults.

### Fix

`collect_mla_module.py` now prepends the OOT vendor dir (computed
from `vllm_ascend.__file__`, not the broken `set_env.bash` that
hardcodes a non-pip-install path) to `ASCEND_CUSTOM_OPP_PATH`
**before** `import torch`:

```python
# in collect_mla_module.py module preamble, before any torch import
def _ensure_oot_custom_opp_path() -> None:
    import vllm_ascend
    oot = os.path.join(os.path.dirname(vllm_ascend.__file__),
                       "_cann_ops_custom", "vendors", "vllm-ascend")
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{oot}:{existing}" if existing else oot)

_ensure_oot_custom_opp_path()
import torch
```

Set `AIC_SKIP_OOT_PATH_FIX=1` to opt out (e.g. if you've already
exported the path in the shell).

### Verified working configurations

| batch | seq_len | mode | latency |
|---:|---:|---|---:|
| 1 | 512 | context | 1.94 ms |
| 4 | 2048 | context | 38.49 ms |

### Sweep status

`dsa_context_module_perf.txt` and `dsa_generation_module_perf.txt`
are being filled by

```sh
python collector/npu/collect_mla_module.py --mode context    --resume \
  --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
python collector/npu/collect_mla_module.py --mode generation --resume \
  --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
```

`--resume` makes the run idempotent against the existing perf .txt,
so a 2-3 day sweep that crashes mid-way can be re-launched without
re-running already-collected (batch, seq) points.

### Historical: bisection that pointed at head_dim>128 (incorrect)

The earlier hypothesis was that CANN 8.5.0 OPP only prebuilt the
`head_dim=128` MLA shape. That was a misreading: bisecting head_dim
in `npu_fused_infer_attention_score` (FIA, a different op) showed
`head_dim ∈ {144, 160, 176, 192, 256}` failing with ACL 561002. That
behaviour is real for FIA, but FIA is not on the collector's
critical path for GLM-5 — DSA goes through SFA. After the OOT
dispatch fix, SFA runs at `head_dim=192/256` (GLM-5's MLA shape)
without trouble. The FIA-specific 561002 result is preserved here
in case anyone re-uses these dimensions outside the SFA path.

Diagnostic tooling kept for reuse:

- `tools/probe_sfa_standalone.sh` — minimal SFA reproducer
- `tools/probe_sfa_full_error.sh` — capture full ACL error
- `tools/inspect_sfa_binaries.sh` — dump prebuilt OPP bin metadata
- `tools/inspect_oot_sfa_bins.sh` — compare OOT vs CANN-shipped bins
- `tools/run_collect_with_plog.sh` — capture CANN plog around a
  collector run (vllm-ascend rejects ASCEND_LAUNCH_BLOCKING=1, so
  plog is the supported way to get accurate kernel-level errors)
- `tools/diag_sfa_registration.sh` — reproduce the OOT registration
  problem in isolation
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

# DSA SFA / MLA module-level data (GLM-5, head_dim=192/256)
python collector/npu/collect_mla_module.py --mode context --resume \
  --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
python collector/npu/collect_mla_module.py --mode generation --resume \
  --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
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
