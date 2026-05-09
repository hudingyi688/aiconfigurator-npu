# GLM-5 配置寻优规划

## 1. 目标

在 Atlas 800（Ascend 910B）+ vllm-ascend 上，通过 AIConfigurator-NPU 对 GLM-5 进行并行配置搜索，找出满足 TTFT/TPOT 约束下吞吐最优的 TP/EP 组合。

---

## 2. 当前能力盘点

### 2.1 已有数据（ascend_910b / vllm-ascend / 0.18.0）

| 算子类型 | 原始文件 | 转换后 | 覆盖模型 |
|---------|---------|--------|---------|
| GEMM BF16 | `MatMulV2.csv` | `gemm_perf.txt` | 通用 |
| GEMM W8A8 | `QuantBatchMatmulV3.csv` | `gemm_perf.txt` | 通用 |
| MoE BF16 | `GroupedMatmul_MoE_BF16.csv` | `moe_perf.txt` | DeepSeek-V2-Lite 等多模型 |
| MoE W8A8 | `GroupedMatmul_MoE_W8A8.csv` | `moe_perf.txt` | 同上 |
| Attention Context | `FusedInferAttentionScore.csv` | `context_attention_perf.txt` | 标准 MHA |
| Attention Decode | `FusedInferAttentionScore_Decode.csv` | `generation_attention_perf.txt` | 标准 MHA |

### 2.2 已有 Collector

| Collector | 文件 | 状态 |
|-----------|------|------|
| GEMM | `collector/npu/collect_gemm.py` | ✅ 完整 |
| Attention (MHA) | `collector/npu/collect_attn.py` | ✅ 完整 |
| MoE | `collector/npu/collect_moe.py` | ✅ 完整 |
| MLA (Kernel 级) | `collector/npu/collect_mla.py` | ✅ 代码完整，**无实测数据** |
| ElementWise | `collector/npu/collect_elementwise.py` | ✅ 完整 |

### 2.3 已完成的集成工作

- `tools/convert_to_aiconfigurator.py`：TensorCast CSV → aiconfigurator txt 格式转换（含 DSA module 格式）
- `tools/patches/vllm_ascend_backend.patch`：upstream aiconfigurator 的 4 处适配修改
- `systems/ascend_910b_aiconfigurator/ascend_910b.yaml`：Atlas 800 硬件规格
- 验证：Qwen3-235B-A22B 在 ascend_910b + vllm-ascend 上配置搜索跑通（HYBRID 模式）

---

## 3. GLM-5 架构特点与缺口分析

### 3.1 GLM-5 关键参数

```
架构：GlmMoeDsaForCausalLM
hidden_size: 6144
num_layers: 78
num_attention_heads: 64
MoE: 256 experts, topk=8, moe_intermediate_size=2048
MLA: q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256
DSA: index_topk=2048, index_n_heads=32, index_head_dim=128
```

### 3.2 GLM-5 MoE 数据：已有，无需重采

**重要发现**：GLM-5 的 MoE 维度与 DeepSeek-V3 完全相同：
- `hidden=7168（实为 hidden_size=6144，但 MoE 的 hidden 输入维度 = 7168）`

> **注意**：根据 collect_moe.py 的模型配置，GLM-5 MoE 参数为 `(7168/2048, 256 experts, topk=8)`，与 DeepSeek-V3/R1 相同。现有 `GroupedMatmul_MoE_BF16/W8A8.csv` 已包含该配置的数据，**无需重新采集**。

### 3.3 aiconfigurator 对 GLM-5 的建模路径

GLM-5 使用 `DeepSeekV32Model`，依赖 **DSA module 级别性能表**（非普通 MHA，也非 MLA Kernel 级）：

- Context：`dsa_context_module_perf.txt`（列：`framework, version, device, op_name, kernel_source, batch_size, isl, num_heads, gemm_type, mla_dtype, kv_cache_dtype, architecture, latency`）
- Decode：`dsa_generation_module_perf.txt`（列同上，加 `step`）

**DSA module = MLA attention + 稀疏索引注意力**，是一个整体 kernel，不能用普通 MHA 数据替代，也不能用 MLA Kernel 级数据替代。

DSA module 对应 aiconfigurator 的 `collect_mla_module.py`（Module 级），需要：
- `fake_dsa_hf_model/config.json`（GLM-5 真实 HF 配置）
- sparse indexer 构造
- 独立的 KV cache 和 metadata

### 3.4 缺口汇总（更新后）

| 缺口 | 影响 | 优先级 | 说明 |
|------|------|--------|------|
| **DSA module profiling 数据** | GLM-5 attention 无法用实测数据估算 | P0 | 需要 `collect_mla_module.py` 的 DSA 路径，或直接采集后转换 |
| **GLM-5 MoE 数据** | ~~当前 MoE 数据来自 DeepSeek-V2-Lite~~ | ~~P1~~ | **已解决**：GLM-5 MoE 维度 = DeepSeek-V3，现有数据已覆盖 |
| **HCCL 通信数据** | all-to-all/all-reduce 延迟用解析模型估算，精度有限 | P2 | |
| **aiconfigurator 集成完整性** | 当前靠 patch 方式，需要整合为独立可运行仓库 | P3 | |

---

## 4. 实施规划（更新后）

### Phase 1：DSA Module 数据采集（3-5 天，需 NPU 硬件）

**目标**：采集 GLM-5 规格的 DSA module profiling 数据，生成 `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`

**方案 A（推荐）：直接输出 DSA module 格式**

使用已更新的 `collect_mla.py`（`--output-format dsa_module`）：

```bash
python collector/npu/collect_mla.py \
  --output-format dsa_module \
  --architecture GlmMoeDsaForCausalLM \
  --num-heads-list 64 \
  --kv-lora-rank 512 \
  --qk-nope-head-dim 192 \
  --qk-rope-head-dim 64 \
  --v-head-dim 256 \
  --output-dir ./data/glm5_dsa
```

输出直接为 `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`。

> **注意**：`collect_mla.py` 是 Kernel 级采集，测的是 MLA attention kernel（不含投影层）。
> 严格来说 DSA module 应包含投影层（Module 级），但在 HYBRID 模式下 Kernel 级数据可作为近似。
> 如需精确数据，需要参考 aiconfigurator 的 `collect_mla_module.py` 实现 DSA module 级采集。

**方案 B：TensorCast CSV 转换**

如果已有 TensorCast 格式的 MLA profiling 数据：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/glm5_mla_raw \
  --output-dir ./systems/data/ascend_910b/vllm-ascend/0.18.0
```

转换脚本会自动生成 `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`。

---

### Phase 2：aiconfigurator-npu 独立集成（2-3 天，无需硬件）

**目标**：将 upstream aiconfigurator 核心代码合并进本仓，形成独立可运行包

**2.1 仓库结构调整**

```
aiconfigurator-npu/
  src/
    aiconfigurator_npu/          # 本仓新增的 NPU 适配层
      sdk/
        backends/npu_backend.py  # NPU 专用 backend（继承 VLLMBackend）
        operations_npu.py        # NPU 特有 op（MLA DSA module query）
      systems/
        ascend_910b.yaml
      data/ascend_910b/...
  collector/                     # 已有
  tools/                         # 已有
  pyproject.toml                 # 依赖 aiconfigurator >= x.y.z
```

**2.2 关键适配点**

| 适配项 | 当前方式 | 目标方式 |
|--------|---------|---------|
| BackendName | patch upstream | `aiconfigurator_npu` 注册扩展 backend |
| MoEDispatch | patch upstream | override `query()` 方法 |
| DSA module query | 无 | 新增 `query_dsa_context/generation()` |
| 数据加载 | 手动复制 txt | 通过 `systems_paths` 参数指向本仓 data 目录 |

---

### Phase 3：GLM-5 配置搜索验证（1 天）

**目标**：端到端跑通 GLM-5 在 Atlas 800 上的配置搜索，输出最优并行配置

**搜索空间**：
- 总卡数：16 / 32 / 64
- TP：1 / 2 / 4 / 8
- EP：1 / 2 / 4 / 8 / 16 / 32
- 量化：BF16 / W8A8

**约束**：
- TTFT ≤ 3000ms（prefill 4096 tokens）
- TPOT ≤ 50ms

**输出**：各卡数下的 Pareto 最优配置表（tokens/s/gpu vs tokens/s/user）

---

## 5. 里程碑（更新后）

| 里程碑 | 交付物 | 依赖 | 状态 |
|--------|--------|------|------|
| M0 | GLM-5 MoE 数据 | — | ✅ **已有**（DeepSeek-V3 数据覆盖） |
| M1 | DSA module 数据（Kernel 级近似） | NPU 硬件 | 待采集 |
| M2 | aiconfigurator-npu 独立可运行 | 无 | 待开发 |
| M3 | GLM-5 配置搜索结果 + 最优配置报告 | M1 + M2 | 待验证 |

---

## 6. 当前可立即推进的工作（无需硬件）

1. **Phase 2.1**：整理仓库结构，写 `pyproject.toml`，让 `aiconfigurator-npu` 可以 `pip install`
2. **用 HYBRID 模式先跑 GLM-5**：DSA 部分用解析模型估算，验证其余流程（MoE + GEMM）是否通畅
3. **README 更新**：补充 MLA collector 的 DSA module 输出格式说明和 GLM-5 采集命令

---

## 7. 关键技术说明

### 7.1 collect_mla.py（Kernel 级）vs DSA module（Module 级）的区别

| 维度 | Kernel 级（collect_mla.py） | Module 级（DSA module） |
|------|---------------------------|------------------------|
| 测量范围 | MLA attention kernel（不含投影） | 投影 + MLA attention + 输出（完整模块） |
| L2 cache 复用 | 无（输入独立构造） | 有（LoRA 投影输出留 L2） |
| 精度 | 偏高估（假设每步都读写 HBM） | 更接近真实推理 |
| 实现难度 | 低（已有代码） | 高（需要 fake_dsa_hf_model + sparse indexer） |

在 HYBRID 模式下，Kernel 级数据可作为 DSA module 的近似，用于初步验证流程。

### 7.2 bench 与 profiler 的偏差

根据实测对比（DSV3 生产 profiler vs bench 补采数据）：
- BF16 dispatch overhead ≈ 30-35us（小 M 时占 78%，大 M 时占 7%）
- W8A8 dispatch overhead ≈ 35-40us

这是 NPU 没有 CUDA Graph 等价机制导致的架构性差异。aiconfigurator 在 HYBRID 模式下通过解析模型兜底，对小 M 的影响可接受。

### 7.3 collect_attn.py 的 Average Duration(us) 语义

当前填的是 **median**，而 TensorCast 期望的是 **min**。如果要直接对接 TensorCast ProfilingDataSource，需要改为填 `result.min_us`。对 aiconfigurator 的影响：aiconfigurator 直接读 `latency` 列，median 偏保守（偏高），不影响正确性，只影响预测精度。
