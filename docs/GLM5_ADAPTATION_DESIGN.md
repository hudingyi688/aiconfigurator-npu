# GLM-5 适配设计文档

基于 AIConfigurator-NPU 对 GLM-5（GlmMoeDsaForCausalLM）的完整适配记录与设计说明。

创建日期：2026-05-09
最后更新：2026-05-11

---

## 0. 适配进展摘要

| 阶段 | 状态 | 说明 |
|------|------|------|
| 架构分析 | ✅ 完成 | GLM-5 算子类型、维度梳理 |
| MoE 数据 | ✅ 完成 | DeepSeek-V3 数据覆盖，无需重采 |
| Backend Patch | ✅ 完成 | `vllm_ascend_backend.patch` 5 处改动 |
| 系统规格 | ✅ 完成 | `ascend_910b.yaml` 硬件配置 |
| DSA Collector | ✅ 完成 | `collect_mla_module.py` Module 级采集脚本 |
| DSA 数据采集 | ⬜ 待进行 | 需 NPU 硬件执行采集 |
| 配置搜索验证 | ⬜ 待进行 | 待 DSA 数据完成后验证 |

**下一步**：在 NPU 硬件上执行 `collect_mla_module.py`，采集 DSA Module 性能数据。

---

## 1. GLM-5 架构概览

```
架构类名：GlmMoeDsaForCausalLM
hidden_size:          6144
num_layers:           78
num_attention_heads:  64

MoE 参数：
  num_experts:          256
  topk:                 8
  moe_intermediate_size: 2048
  （注：MoE hidden 输入维度 = 7168，与 DeepSeek-V3 相同）

MLA 参数：
  q_lora_rank:          2048
  kv_lora_rank:         512
  qk_nope_head_dim:     192
  qk_rope_head_dim:     64
  v_head_dim:           256

DSA（DeepSeek Sparse Attention）参数：
  index_topk:           2048
  index_n_heads:        32
  index_head_dim:       128
```

GLM-5 使用 DSA（DeepSeek Sparse Attention）——在 MLA 基础上叠加稀疏索引注意力，每个 token 只 attend 到部分 KV position，减少 attention 计算量但增加 sparse indexer 开销。

### 1.2 完整算子类型

算子类型由模型 config 决定存在性，由 vLLM/vllm-ascend 决定具体实现路径（见第 10 节）。

| 算子 | 维度 | 层数 | 作用 | bench 脚本 |
|------|------|------|------|-----------|
| **Embedding** | vocab=154880 → hidden=6144 | 1 | 将 token id 映射为 hidden_size 向量，推理时只在 prefill 首 token 执行 | 未采集（延迟可忽略） |
| **Dense MLP** | 6144 → 12288×2 → 6144 | 3 层（layer 0-2） | `first_k_dense_replace=3`，前 3 层用标准 FFN（gate_up_proj + SiLU + down_proj），`intermediate_size=12288` | `collect_gemm.py`（GEMM BF16/W8A8） |
| **MoE routed experts** | 6144 → 2048×2 → 6144，256 experts topk=8 | 75 层（layer 3-77） | `moe_layer_freq=1`，第 3 层起全部是 MoE；每个 token 路由到 8 个 expert，GroupedGEMM 并行计算 | `collect_moe.py`（MoE BF16/W8A8） |
| **MoE shared expert** | 6144 → 2048×2 → 6144，1 expert | 75 层 | `n_shared_experts=1`，每个 MoE 层有 1 个 shared expert，所有 token 都经过，与 routed experts 并行执行后相加 | `collect_gemm.py`（维度同 Dense MLP，M=batch） |
| **DSA Attention** | 见 MLA 参数 | 全部 78 层 | 完整 MLA 模块：fused_qkv_a_proj → q_a_layernorm → q_b_proj → kv_a_layernorm → kv_b_proj → sparse attention → o_proj；每个 token 只 attend 到 index_topk=2048 个 KV position | `collect_mla_module.py`（Method C，Module 级） |
| **RMSNorm** | hidden=6144 | 78×2 + 1 = 157 | 每层的 input_layernorm（attention 前）+ post_attention_layernorm（FFN 前）+ 最终 norm；实际走 fused Add+RMSNorm（含残差加法） | `collect_elementwise.py`（rmsnorm / add_rmsnorm） |
| **LM Head** | 6144 → 154880 | 1 | 将最后一层 hidden state 投影到词表，取 argmax 得到下一个 token | `collect_gemm.py`（GEMM BF16，M=batch，N=154880，K=6144） |
| **AllReduce** | hidden=6144 | 每层（TP>1 时） | TP 模式下 attention o_proj 和 MLP down_proj 后的 all-reduce，合并各 TP rank 的部分结果 | 未采集（aiconfigurator 用解析模型估算） |
| **AllToAll** | token dispatch/combine | 每 MoE 层（EP>1 时） | EP 模式下 MoE dispatch（将 token 发送到对应 expert 所在 rank）和 combine（收集 expert 输出），是 MoE 延迟的主要瓶颈之一 | 未采集（aiconfigurator 用解析模型估算） |

**bench 脚本覆盖情况**：

| bench 脚本 | 覆盖算子 | 状态 |
|-----------|---------|------|
| `collect_gemm.py` | Dense MLP、shared expert、LM Head 的线性层 | ✅ 有实测数据 |
| `collect_moe.py` | MoE routed experts（GroupedGEMM） | ✅ 有实测数据（DeepSeek-V3 维度覆盖 GLM-5） |
| `collect_mla_module.py` | DSA Attention（完整 Module 级） | ⬜ 待采集（需 NPU 硬件） |
| `collect_elementwise.py` | RMSNorm（rmsnorm / add_rmsnorm） | ✅ 有实测数据 |
| `collect_attn.py` | 标准 MHA attention（非 DSA，GLM-5 不用） | — 不适用 |
| `collect_mla.py` | MLA attention kernel 级（Kernel 级近似） | ⬜ 无实测数据（可作为 DSA 近似） |
| 无 | AllReduce、AllToAll、Embedding | — aiconfigurator 用解析模型估算 |

---

## 2. aiconfigurator 对 GLM-5 的建模路径

aiconfigurator 将 GLM-5 映射到 `DeepSeekV32Model`，其 attention 部分依赖 **DSA module 级别性能表**，而非普通 MHA 或 MLA Kernel 级数据。

### 2.1 性能表格式

**Context（Prefill）**：`dsa_context_module_perf.txt`

| 列名 | 说明 |
|------|------|
| framework | `vllm-ascend` |
| version | `0.18.0` |
| device | `Ascend 910B` |
| op_name | `dsa_context_module` |
| kernel_source | `vllm_ascend_mla` |
| batch_size | batch 大小 |
| isl | 输入序列长度 |
| num_heads | attention head 数（GLM-5 = 64） |
| gemm_type | GEMM 量化模式（`float16` / `sq` / `w8a8_dynamic`） |
| mla_dtype | MLA/FMHA 量化模式（`float16` / `fp8`） |
| kv_cache_dtype | KV cache 量化模式（`float16` / `int8` / `fp8`） |
| architecture | `GlmMoeDsaForCausalLM` |
| latency | 延迟（ms） |

**Generation（Decode）**：`dsa_generation_module_perf.txt`

同上，额外增加 `step` 列（`s = isl + step`，即总 context 长度）。

### 2.2 数据加载路径

aiconfigurator 的 `perf_database.py` 通过 `load_context_dsa_module_data()` / `load_generation_dsa_module_data()` 加载，索引结构为：

```
context:    data[fmha_mode][kv_cache_mode][gemm_mode][architecture][num_heads][isl][batch]
generation: data[kv_cache_mode][gemm_mode][architecture][num_heads][batch][isl+step]
```

`architecture = "GlmMoeDsaForCausalLM"` 对应 `DSA_MODEL_DIMS` 中的 GLM-5 维度配置（已内置于 aiconfigurator）。

---

## 3. MoE 数据：已有，无需重采

**关键发现**：GLM-5 的 MoE 维度与 DeepSeek-V3/R1 完全相同：

| 参数 | GLM-5 | DeepSeek-V3 |
|------|-------|-------------|
| MoE hidden 输入 | 7168 | 7168 |
| intermediate_size | 2048 | 2048 |
| num_experts | 256 | 256 |
| topk | 8 | 8 |

现有 `GroupedMatmul_MoE_BF16.csv` / `GroupedMatmul_MoE_W8A8.csv` 已包含该配置，转换后的 `moe_perf.txt` 可直接用于 GLM-5 配置搜索。

---

## 4. DSA Module 数据采集方案

### 4.1 三种方案的调用层级对比

三种方案的本质区别在于**调用的 vllm-ascend 接口层级不同**：

```
方案 A（Kernel 级近似）— 调 vllm-ascend MLA impl 内部方法：
  collect_mla.py
    → mla_factory.py
      → AscendMLAImpl._forward_prefill() / _forward_decode()
        → torch_npu.npu_fused_infer_attention_score_v2()  ← CANN kernel
  测量范围：MLA attention kernel（不含投影层 W_q_a, W_q_b, W_kv_a）

方案 B（TensorCast 转换）— 直调 torch_npu kernel（最底层）：
  op_replay/FusedInferAttentionScore_MLA_run.py（TensorCast 体系）
    → torch_npu.npu_fused_infer_attention_score()  ← 直调 CANN kernel
  测量范围：纯 CANN kernel（不含任何框架层开销）
  注：方案 B 的 CSV 来自 TensorCast msprof 外部挂载，不是本仓采集

方案 C（Module 级精确）— 调完整 vllm-ascend 模块（最高层）：
  collect_mla_module.py（已实现）
    → DeepseekV2MLAAttention.forward()  ← vllm-ascend 完整模块
      → W_q_a → RmsNorm → W_q_b → Q
      → W_kv_a → kv_a_layernorm → kv_c + k_pe
      → AscendSFAImpl.forward()  ← DSA backend（use_sparse=True）
        → indexer_select_pre_process / post_process
        → npu_sparse_flash_attention
      → W_o → output
  测量范围：投影 + DSA attention + 输出（含层间 L2 cache 复用）
```

**方案 A 与 B 的关键区别**：方案 A 走 vllm-ascend 的 `AscendMLAImpl` 框架层（含 metadata 构造、状态机 dispatch），方案 B 直调 `torch_npu` kernel（跳过所有框架层）。两者最终都落到同一个 CANN kernel，但方案 A 包含 ~30-35us 的框架 dispatch overhead。

### 4.2 方案对比

| 维度 | 方案 A（Kernel 级近似） | 方案 B（TensorCast 转换） | 方案 C（Module 级精确） |
|------|----------------------|------------------------|------------------------|
| 调用层级 | vllm-ascend MLA impl 内部方法 | torch_npu 直调 kernel | vllm-ascend 完整模块 |
| 测量范围 | MLA attention kernel（不含投影） | 纯 CANN kernel | 投影 + attention + 输出 |
| 框架 dispatch overhead | 含（~30-35us） | 不含 | 含（更多，含投影层） |
| L2 cache 复用 | 无 | 无 | 有（LoRA 投影输出留 L2） |
| 精度 | 中（高估，假设每步读写 HBM） | 偏乐观（纯 kernel，无 dispatch） | 最高（最接近真实推理） |
| 数据来源 | 本仓 collect_mla.py 采集 | TensorCast msprof 外部挂载 | 待实现 |
| 实现难度 | 低（代码已有） | 低（已有转换脚本） | 高（需要 fake_dsa_hf_model + sparse indexer） |
| 当前状态 | 代码完整，无实测数据 | 依赖 TensorCast 已有数据 | **已实现**（`collect_mla_module.py`） |

**结论**：在 HYBRID 模式下，方案 A 或 B 均可作为 DSA module 的近似，用于初步验证流程。精确数据需要方案 C。

### 4.3 方案 A：Kernel 级近似（推荐用于流程验证）

使用已更新的 `collect_mla.py`，通过 `AscendMLAImpl._forward_prefill()` / `_forward_decode()` 采集，直接输出 DSA module 格式：

```bash
python collector/npu/collect_mla.py \
  --output-format dsa_module \
  --architecture GlmMoeDsaForCausalLM \
  --num-heads-list 64 \
  --kv-lora-rank 512 \
  --qk-nope-head-dim 192 \
  --qk-rope-head-dim 64 \
  --v-head-dim 256 \
  --framework vllm-ascend \
  --version 0.18.0 \
  --device "Ascend 910B" \
  --output-dir ./data/glm5_dsa
```

输出：
- `data/glm5_dsa/dsa_context_module_perf.txt`
- `data/glm5_dsa/dsa_generation_module_perf.txt`

然后复制到数据目录：

```bash
cp data/glm5_dsa/dsa_context_module_perf.txt \
   systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/glm5_dsa/dsa_generation_module_perf.txt \
   systems/data/ascend_910b/vllm-ascend/0.18.0/
```

### 4.4 方案 B：TensorCast CSV 转换

如果已有 TensorCast msprof 采集的 MLA profiling 数据（`FusedInferAttentionScore_MLA.csv` / `FusedInferAttentionScore_Decode_MLA.csv`），使用转换脚本：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/glm5_mla_raw \
  --output-dir ./systems/data/ascend_910b/vllm-ascend/0.18.0 \
  --device "Ascend 910B" \
  --framework vllm-ascend \
  --version 0.18.0
```

转换脚本会自动读取 CSV 中的 `Architecture` 列（`GlmMoeDsaForCausalLM`），生成正确的 DSA module 格式文件。

### 4.5 方案 C：Module 级精确采集（已实现）

`collector/npu/collect_mla_module.py` 实现了完整的 Module 级采集，调用链：

```
collect_mla_module.py
  → DeepseekV2MLAAttention.forward()
    → W_q_a → RmsNorm → W_q_b → Q
    → W_kv_a → kv_a_layernorm → kv_c + k_pe
    → AscendSFAImpl.forward()   ← DSA backend (use_sparse=True)
      → indexer_select_pre_process / post_process
      → npu_sparse_flash_attention
    → W_o → output
```

**使用方式**：

```bash
# Context（Prefill）
python collector/npu/collect_mla_module.py \
  --mode context \
  --model zai-org/GLM-5 \
  --output-dir ./data/glm5_dsa_module

# Generation（Decode）
python collector/npu/collect_mla_module.py \
  --mode generation \
  --model zai-org/GLM-5 \
  --output-dir ./data/glm5_dsa_module

# 快速单点验证
python collector/npu/collect_mla_module.py \
  --mode context --quick --batch-size 4 --seq-len 2048
```

输出直接为 `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`，可直接复制到数据目录：

```bash
cp data/glm5_dsa_module/dsa_context_module_perf.txt \
   systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/glm5_dsa_module/dsa_generation_module_perf.txt \
   systems/data/ascend_910b/vllm-ascend/0.18.0/
```

**实现要点**：
- 使用 aiconfigurator 内置的 `zai-org--GLM-5_config.json`，无需 HuggingFace Hub 下载
- `_create_npu_vllm_config()` 创建完整 VllmConfig 并调用 `init_ascend_config()`
- DSA indexer KV cache（uint8 格式）在 `hf_config.index_head_dim` 存在时自动创建
- `benchmark_npu()` 优先使用 NPU Graph，失败时回退到 eager 模式

---

## 5. collect_mla.py 适配说明

### 5.1 已完成的更新

`collector/npu/collect_mla.py` 已更新，支持以下新功能：

**新增参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-format` | `tensorcast` | `tensorcast`（TensorCast CSV）或 `dsa_module`（aiconfigurator 格式） |
| `--architecture` | `GlmMoeDsaForCausalLM` | 模型架构标识符 |
| `--framework` | `vllm-ascend` | 写入 dsa_module 输出的 framework 字段 |
| `--version` | `0.18.0` | 写入 dsa_module 输出的 version 字段 |
| `--device` | `Ascend 910B` | 写入 dsa_module 输出的 device 字段 |
| `--gemm-type` | `float16` | GEMM 量化模式（`GEMMQuantMode` 枚举名） |
| `--mla-dtype` | `float16` | MLA 量化模式（`FMHAQuantMode` 枚举名） |
| `--kv-cache-dtype` | `float16` | KV cache 量化模式（`KVCacheQuantMode` 枚举名） |

**默认参数更新**（从 DeepSeek-V3 改为 GLM-5）：

| 参数 | 旧默认值（DSV3） | 新默认值（GLM-5） |
|------|----------------|-----------------|
| `--num-heads-list` | `[128]` | `[64]` |
| `--qk-nope-head-dim` | `128` | `192` |
| `--v-head-dim` | `128` | `256` |

**TensorCast CSV 新增列**：`Architecture`（用于 `convert_to_aiconfigurator.py` 的 DSA module 转换）

### 5.2 输出文件对应关系

| `--output-format` | 输出文件 |
|-------------------|---------|
| `tensorcast` | `FusedInferAttentionScore_MLA.csv`（context）<br>`FusedInferAttentionScore_Decode_MLA.csv`（generation） |
| `dsa_module` | `dsa_context_module_perf.txt`<br>`dsa_generation_module_perf.txt` |

---

## 6. convert_to_aiconfigurator.py 适配说明

### 6.1 新增 convert_dsa_module()

`tools/convert_to_aiconfigurator.py` 新增 `convert_dsa_module()` 函数，自动处理：
- 读取 `FusedInferAttentionScore_MLA.csv`（context）和 `FusedInferAttentionScore_Decode_MLA.csv`（generation）
- 从 CSV 的 `Architecture` 列读取架构标识符（默认 `GlmMoeDsaForCausalLM`）
- 输出 `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`
- Generation 格式：`isl=1, step=seq_len-1`（总 context = seq_len）

---

## 7. 配置搜索流程

### 7.0 aiconfigurator NPU 移植状态

配置搜索依赖 upstream aiconfigurator，需要先完成移植。移植方式为 patch + 数据复制，通过 `tools/apply_patches.sh` 一键完成。

**已完成的 patch（`tools/patches/vllm_ascend_backend.patch`，共 5 个文件）**：

| 文件 | 改动 | 原因 |
|------|------|------|
| `sdk/common.py` | 新增 `BackendName.vllm_ascend = "vllm-ascend"`；新增 `GEMMQuantMode.w8a8_dynamic`、`MoEQuantMode.w8a8_dynamic` | NPU 量化模式注册 |
| `sdk/backends/factory.py` | `get_backend()` 支持 `vllm_ascend` → 映射到 `VLLMBackend` | 后端路由 |
| `sdk/operations.py` | `MoEDispatch.query()` 的 vllm 分支扩展到 `vllm_ascend` | MoE dispatch 延迟查询 |
| `sdk/perf_database.py` | `supported_quant_mode` 初始化分支扩展到 `vllm_ascend`；MoE 查询分支扩展到 `vllm_ascend` | 数据库初始化 + MoE 查询路径 |
| `sdk/task.py` | `build_disagg_parallel_lists()` 和 `TaskConfigFactory` 的 vllm 分支扩展到 `vllm_ascend` | 并行配置生成 |

**移植步骤**：

```bash
# 1. 克隆 upstream aiconfigurator（如未克隆）
git clone https://github.com/ai-dynamo/aiconfigurator /path/to/aiconfigurator

# 2. 应用 patch + 复制数据
cd /path/to/aiconfigurator-npu
./tools/apply_patches.sh /path/to/aiconfigurator

# 3. 安装
pip install -e /path/to/aiconfigurator
```

`apply_patches.sh` 会自动：
- 应用 `vllm_ascend_backend.patch`（幂等，已应用则跳过）
- 复制 `systems/data/ascend_910b/` → upstream 的 `systems/data/ascend_910b/`
- 复制 `systems/ascend_910b_aiconfigurator/ascend_910b.yaml` → upstream 的 `systems/`

### 7.1 数据准备检查清单

| 数据文件 | 状态 | 说明 |
|---------|------|------|
| `gemm_perf.txt` | ✅ 已有 | MatMulV2 + QuantBatchMatmulV3 |
| `moe_perf.txt` | ✅ 已有 | DeepSeek-V3 配置覆盖 GLM-5 |
| `context_attention_perf.txt` | ✅ 已有 | 标准 MHA（非 DSA 层用） |
| `generation_attention_perf.txt` | ✅ 已有 | 标准 MHA（非 DSA 层用） |
| `dsa_context_module_perf.txt` | ⬜ 待采集 | `collect_mla_module.py` 已就绪，需 NPU 硬件 |
| `dsa_generation_module_perf.txt` | ⬜ 待采集 | `collect_mla_module.py` 已就绪，需 NPU 硬件 |

### 7.2 配置搜索工作原理

aiconfigurator 的配置搜索是**解析模型 + 实测数据库**的混合估算框架，不需要实际部署模型：

```
TaskConfig（搜索参数）
    ↓
PerfDatabase（加载 perf .txt 数据）
    ↓
Model.context_ops / generation_ops（算子序列）
    ↓  每个算子调用 Operation.query(database, num_tokens, ...)
    ↓  → GEMM: 查 gemm_perf.txt，按 (M,N,K,quant) 插值
    ↓  → MoE:  查 moe_perf.txt，按 (tokens,hidden,inter,topk,ep) 插值
    ↓  → DSA:  查 dsa_context/generation_module_perf.txt，按 (batch,isl,heads) 插值
    ↓  → Comm: 解析模型（AllReduce/AllToAll 用带宽公式估算）
    ↓  → Norm/Embed: 解析模型（roofline）
    ↓
InferenceSummary（TTFT / TPOT / throughput 估算）
    ↓
ParetoAnalysis（过滤满足 SLA 的配置，输出 Pareto 最优集）
```

**HYBRID 模式**：当某类算子没有实测数据时（如 DSA module 数据缺失），自动退回解析模型估算，不会报错。适合在 DSA 数据采集前先验证 MoE + GEMM 路径。

**SILICON 模式**：全部使用实测数据，缺数据则报错。用于最终精确搜索。

### 7.3 执行配置搜索

**Step 1：环境准备**

```bash
# 应用 patch（见 7.0）
./tools/apply_patches.sh /path/to/aiconfigurator
pip install -e /path/to/aiconfigurator

# 如有 DSA 数据，先复制到数据目录
cp data/glm5_dsa_module/dsa_context_module_perf.txt \
   /path/to/aiconfigurator/src/aiconfigurator/systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/glm5_dsa_module/dsa_generation_module_perf.txt \
   /path/to/aiconfigurator/src/aiconfigurator/systems/data/ascend_910b/vllm-ascend/0.18.0/
```

**Step 2：运行搜索**

```python
from aiconfigurator.sdk.task import TaskConfig
from aiconfigurator.sdk.common import DatabaseMode, GEMMQuantMode, MoEQuantMode

# HYBRID 模式（DSA 数据缺失时用解析模型兜底）
task = TaskConfig(
    model="zai-org/GLM-5",
    backend="vllm-ascend",
    system="ascend_910b",
    database_mode=DatabaseMode.HYBRID,
    isl=4096,          # prefill 输入长度
    osl=512,           # 生成长度
    num_requests=100,  # 并发请求数
    gemm_quant_mode=GEMMQuantMode.float16,    # BF16；W8A8 用 w8a8_dynamic
    moe_quant_mode=MoEQuantMode.float16,
)

results = task.run()
results.pareto_analysis(
    ttft_sla_ms=3000,   # TTFT ≤ 3000ms
    tpot_sla_ms=50,     # TPOT ≤ 50ms
)
results.print_pareto_table()
```

**Step 3：解读结果**

输出为各 (total_gpus, TP, EP) 组合下的 TTFT / TPOT / throughput 估算，以及满足 SLA 的 Pareto 最优配置集。

### 7.4 搜索空间

| 参数 | 候选值 |
|------|--------|
| 总卡数 | 16 / 32 / 64 |
| TP | 1 / 2 / 4 / 8 |
| EP | 1 / 2 / 4 / 8 / 16 / 32 |
| 量化 | BF16 / W8A8 |

**SLA 约束**：
- TTFT ≤ 3000ms（prefill 4096 tokens）
- TPOT ≤ 50ms

---

## 8. 已知限制与注意事项

### 8.1 bench 与 profiler 的偏差

根据 DSV3 生产 profiler 对比实测：

| M 规模 | BF16 dispatch overhead | W8A8 dispatch overhead | dispatch 占比 |
|--------|----------------------|----------------------|--------------|
| 小 M（~10us kernel） | ~30-35us | ~35-40us | ~78% |
| 大 M（~500us kernel） | ~30-35us | ~35-40us | ~7% |

NPU 没有 CUDA Graph 等价机制，每次 forward() 都包含完整 Python dispatch。HYBRID 模式通过解析模型兜底，对小 M 的影响可接受。

### 8.2 collect_attn.py 的 Average Duration(us) 语义

当前填的是 **median**，而 TensorCast 期望的是 **min**。对 aiconfigurator 的影响：aiconfigurator 直接读 `latency` 列，median 偏保守（偏高），不影响正确性，只影响预测精度（会略微高估 attention latency）。

### 8.3 W8A8 在小 M 时更慢

实测数据（M=1/128 时 W8A8 比 BF16 慢 ~30-33%）：

| M | BF16 (us) | W8A8 (us) | 加速比 |
|---|-----------|-----------|--------|
| 1 | 62.07 | 85.12 | 0.73x |
| 128 | 65.71 | 97.80 | 0.67x |
| 4096 | 474.91 | 398.51 | 1.19x |

**结论**：decode 阶段（M≤128）不建议开 W8A8，prefill 阶段（M≥2048）开 W8A8 有收益。

### 8.4 MoE 数据的 EP 模拟

现有 MoE 数据通过 `active_expert_range + group_list slice` 模拟 EP 切分，单卡只分配 local experts 权重，使用均匀随机 routing（非 Power Law 分布）。这与真实推理中的 expert 负载不均有差异，但对配置搜索的相对比较影响有限。

---

## 9. 参考资料

- aiconfigurator 源码：`perf_database.py:load_context_dsa_module_data()` / `load_generation_dsa_module_data()`
- aiconfigurator DSA 维度配置：`perf_database.py:DSA_MODEL_DIMS["GlmMoeDsaForCausalLM"]`
- aiconfigurator Module 级采集参考：`collector/vllm/collect_mla_module.py`（DSA 路径）
- NPU 算子采集适配记录：`docs/AIConfigurator Bench — NPU 算子采集适配记录.docx`
- AIConfigurator vLLM Benchmark 分析：`docs/AIConfigurator 基于 vLLM 的 Benchmark 分析.docx`
- 配置搜索规划：`docs/GLM5_CONFIG_SEARCH_PLAN.md`
