---
name: new-model-data-collection
description: Walk through adding a new model (or model family) to aiconfigurator-npu. Maps the model's HF config to the GEMM/MoE/Attention shapes aiconfigurator queries, audits coverage against existing perf .txt, and emits the minimum collect_gemm.py / collect_moe.py / collect_mla_module.py / collect_elementwise.py invocations to fill the gaps. Always uses the existing collectors (no new sweep wrappers); always goes through vllm-ascend ops, never raw CANN.
---

# 新模型数据采集 SOP

## 何时触发

- 用户说要给 aiconfigurator-npu 适配新模型（"加一个 GLM-5"、"支持 Qwen3 MoE"、"为 Llama-Nemotron 跑配置寻优"）
- 已有 perf 表跑配置寻优时大量算子 fallback 到 SOL/empirical，精度不足
- 新装了 vllm-ascend 想把数据 baseline 重做一份

## 核心原则（不要违背）

1. **接口层约束**：所有 collector 必须调 vllm-ascend 算子接口（`AscendRowParallelLinear` / `AscendSFAImpl` / `AscendFusedMoE` 等），**禁止**绕开 vllm-ascend 直接下发 `aclnn` / `torch_npu.npu_*` kernel。直接采 CANN kernel 是 msprof 的工作，不是本仓的工作。
2. **入口唯一**：每类算子只用一个 collector 入口（`collect_gemm.py` / `collect_moe.py` / `collect_mla_module.py` / `collect_elementwise.py`）。新增模型时**不写新 collector**、不写 sweep wrapper 脚本——用 shell 循环驱动现有 CLI 即可。
3. **可恢复**：所有 collector 都支持 `--resume`，必加。

## 执行步骤

### Step 1: 拿模型参数

```bash
python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
keys = ['hidden_size','num_hidden_layers','num_attention_heads','vocab_size',
        'intermediate_size','moe_intermediate_size',
        'num_local_experts','num_experts_per_tok','first_k_dense_replace',
        'q_lora_rank','kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim',
        'index_head_dim','index_topk','index_n_heads',
        'n_shared_experts','max_position_embeddings']
for k in keys:
    if k in c: print(f'  {k:30s} = {c[k]}')
" model_configs/<model_name>--config.json
```

### Step 2: 推导算子 shape（aiconfigurator 实际查询）

参考 `src/aiconfigurator_npu/sdk/models.py` 里对应的 Model 类（`DeepSeekV32Model` / `QwenMoEModel` / `LlamaModel` 等）的 `context_ops` / `generation_ops` 列表，把每条 `ops.GEMM(...)` 的 (N, K) 抽出来。GLM-5 的 21 对 (N, K) 见 `docs/GLM5_ADAPTATION_DESIGN.md`。

GLM-5 / DeepSeek-V32 类模型的 GEMM 集合（**模板**）：

| op | N | K | 备注 |
|---|---|---|---|
| dense_gate_up TP={1,2,4,8} | `2 * intermediate / tp` | `hidden` | 前 `first_k_dense_replace` 层 |
| dense_ffn2 TP={1,2,4,8} | `hidden` | `intermediate / tp` | 同上 |
| shared_gate_up TP={1,2,4,8} | `2 * moe_intermediate / tp` | `hidden` | 后续 MoE 层的 shared expert |
| shared_ffn2 TP={1,2,4,8} | `hidden` | `moe_intermediate / tp` | 同上 |
| router | `num_experts` | `hidden` | TP 不切 |
| logits / lm_head TP={1,2,4,8} | `vocab / tp` | `hidden` | |

MoE GroupedMatmul 一组：`(hidden, moe_intermediate, num_experts, topk)`

DSA module（GLM-5 / DeepSeek-V32）：`num_heads`、`kv_lora_rank`、`qk_nope/rope_head_dim`、`v_head_dim`、`index_head_dim`、`index_topk` 全在 `collect_mla_module.py` 自己读 hf_config。

### Step 3: 对照已有 perf 表做覆盖率审计

```bash
# 看现有 gemm_perf.txt 里出现过的 (N, K) 和 M
python3 -c "
import csv
gemm = list(csv.DictReader(open('systems/data/ascend_910b/vllm-ascend/0.18.0/gemm_perf.txt')))
nk = sorted(set((int(r['n']), int(r['k'])) for r in gemm))
print(f'(N,K) combos: {len(nk)}')
for n,k in nk: print(f'  N={n:7d} K={k:7d}')
m = sorted(set(int(r['m']) for r in gemm))
print(f'M points: {m}')
"

# MoE
python3 -c "
import csv
m = list(csv.DictReader(open('systems/data/ascend_910b/vllm-ascend/0.18.0/moe_perf.txt')))
combos = sorted(set((r['hidden_size'], r['inter_size'], r['num_experts'], r['topk']) for r in m))
for c in combos: print('  hidden=%s inter=%s n_experts=%s topk=%s' % c)
"

# DSA / Attention（如果模型用 DSA）
wc -l systems/data/ascend_910b/vllm-ascend/0.18.0/dsa_*.txt
```

把 Step 2 推导出的所有 (N, K) / MoE 组合 / DSA 维度对照 Step 3 的现有数据，列出 MISS 项。

### Step 4: 按 MISS 项执行最小补采

#### 4.1 GEMM 长方形 (N, K) 补采

**注册模型 + 一条命令搞定**。`collect_gemm.py` 顶部维护一个 `MODEL_GEMM_SHAPES` dict，新模型在那里加一项就够；NPU 上跑 `--model <name>` 自动 sweep 它注册的所有 (N, K)（非笛卡尔积）。

**Step 4.1.1 — 在 `collector/npu/collect_gemm.py` 顶部 `MODEL_GEMM_SHAPES` 加新条目**

```python
MODEL_GEMM_SHAPES: dict[str, list[tuple[int, int]]] = {
    "GlmMoeDsa": [
        (24576, 6144), (6144, 12288), (4096, 6144), (6144, 2048),
        # ... 20 unique (N, K) pairs ...
    ],
    "<NewModel>": [
        # 把 Step 2 推导出的 (N, K) 集合（去重后）填进来
    ],
}
```

注册名约定：跟 hf_config 里 `architectures[0]` 取一致前缀（GLM-5 的 `GlmMoeDsaForCausalLM` -> `GlmMoeDsa`）。

**Step 4.1.2 — NPU 上一条命令跑 sweep**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_gemm.py \
        --model <NewModel> \
        --output-dir ./data/gemm \
        --resume \
        --quant-types bf16 w8a8_dynamic
```

GLM-5 实例（20 对，BF16+W8A8 ≈ 600 行，30-90 min）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_gemm.py \
        --model GlmMoeDsa --output-dir ./data/gemm --resume \
        --quant-types bf16 w8a8_dynamic
```

**禁止**新写 `collect_<model>_gemm_sweep.sh` 这种 wrapper——`--model` flag 已经把"按模型采"的语义放进 collector 自己。

#### 4.2 MoE GroupedMatmul 补采

如果新模型的 `(hidden, moe_intermediate, num_experts, topk)` 组合不在现有表里：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_moe.py \
        --output-dir ./data/moe --resume \
        --quant-types bf16 w8a8_dynamic \
        --hidden-size <h> \
        --inter-size <moe_inter> \
        --num-experts <n_experts> \
        --topk <topk>
```

#### 4.3 DSA module 补采（GLM-5 / DSV3.2 类）

参数从模型 hf_config 自动读，collector 不需要传维度：

```bash
# 单点 quick smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_mla_module.py \
        --mode context --quick --batch-size 1 --seq-len 512 \
        --model <hf_id> --output-dir ./data/<model>_dsa

# 全量 sweep（context + generation 各 2-3 天）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    nohup python3 collector/npu/collect_mla_module.py \
        --mode context --resume --model <hf_id> \
        --output-dir ./data/<model>_dsa > /tmp/ctx.log 2>&1 &

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    nohup python3 collector/npu/collect_mla_module.py \
        --mode generation --resume --model <hf_id> \
        --output-dir ./data/<model>_dsa > /tmp/gen.log 2>&1 &
```

#### 4.4 ElementWise 补采（如果 `hidden_size` 不在 389 行已采列表里）

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_elementwise.py \
        --output-dir ./data/elementwise --resume \
        --hidden-size-list <h1> <h2> ...
```

#### 4.5 标准 MHA Attention（**仅非 DSA 模型**）

GLM-5 / DSV3.2 不需要这步，它们走 DSA module。Llama-class / Qwen2 走标准 MHA：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector \
    python3 collector/npu/collect_attn.py --mode context --resume \
        --output-dir ./data/attn --num-heads-list <heads> --num-kv-heads-list <kv_heads>
```

### Step 5: 数据落到 systems 目录 + 装包

```bash
# 把采集到的 csv / txt 复制到 aiconfigurator 读的位置（两处都要放）
cp data/gemm/*.csv  systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/<model>_dsa/dsa_*.txt  systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/gemm/*.csv  src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/<model>_dsa/dsa_*.txt  src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/

# 重装让 package_data 生效
pip install -e .
```

CSV 转换成 aiconfigurator 期望的 perf .txt 格式，用现有转换脚本：

```bash
python3 tools/convert_to_lookup_table.py \
    --input-root data \
    --output-root systems/data \
    --device ascend_910_93 --framework vllm-ascend --version 0.18.0
```

### Step 6: 验证覆盖

重跑 Step 3 的覆盖率脚本，确认 MISS 项变 HIT。然后跑配置寻优 smoke：

```bash
aic-npu default --model-path <hf_id> --backend vllm-ascend \
    --system ascend_910b --backend-version 0.18.0 \
    --database-mode SILICON \
    --total-gpus 64 --isl 4096 --osl 512 --ttft 10000 --tpot 200 \
    --save-dir /tmp/<model>_search
```

`--database-mode SILICON` 严格只用实测数据，缺数据立刻报错。能跑通说明覆盖率合格，再切 HYBRID 出最终 Pareto 表。

## 输出报告（必产）

每次走完 SOP，输出一份 markdown 表格：

```markdown
# <model_name> 数据采集报告

## 已有数据命中
- GEMM: <hit_count> / <total_count> (N,K) 对
- MoE: <模型组合是否命中>
- ElementWise: <hidden_size 是否覆盖>
- DSA Attention: <已采点数>

## 补采项（按时间顺序）
| 算子 | 命令 | 行数 | 耗时 | 状态 |
|---|---|---|---|---|
| GEMM (N=4096 K=6144) | collect_gemm --n-list 4096 --k-list 6144 | 30 | 5min | OK |
...

## 验证
- SILICON 模式跑配置寻优是否通过：YES/NO
- 哪些 (TP, EP) 组合仍 fallback：列出
```

## 反模式（看到立刻拒绝）

- ❌ 写新的 `collect_<model>_*.py` 入口
- ❌ 写新的 `collect_<model>_sweep.sh` wrapper（shell 循环驱动现有 CLI 即可）
- ❌ 调 `aclnn` / `torch_npu.npu_*` 跳过 vllm-ascend
- ❌ 改 collector 的 spec 列表写死新模型 shape（除非真的是把 spec 列表重构成"按模型注册"的全局重构）
- ❌ 用 msprof 采 kernel 数据混进 perf.txt（profiler 数据另有用途，参考 `comm-alignment` skill）
- ❌ 用解析模型估算硬充实测数据
