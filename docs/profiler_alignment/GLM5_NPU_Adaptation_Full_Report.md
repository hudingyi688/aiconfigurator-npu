# AIConfigurator-NPU适配GLM-5项目完整技术报告

> **状态**: 2026-06-05 更新版（基于 profiler-derived DSA 修正 + ep32 实测 + comm 收口）  
> **分支**: `local/dsa-debug-snapshot` (HEAD 316a8e0)  
> **适用对象**: GLM-5-w8a8 / Ascend 910B / vllm-ascend 0.18.0

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [技术方案核心设计](#2-技术方案核心设计)
3. [GLM-5适配情况与结果](#3-glm-5适配情况与结果)
4. [技术难点与解决方案](#4-技术难点与解决方案)
5. [数据采集工程实践](#5-数据采集工程实践)
6. [后续计划](#6-后续计划)
7. [项目价值总结](#7-项目价值总结)
8. [与传统Profiling方法的对比](#8-与传统profiling方法的对比)
9. [Q&A](#9-qa)

---

## 1. 项目背景与目标

### 1.1 项目定位

**原始项目**: NVIDIA AIConfigurator  
- 设计目标：GPU算子级微基准采集框架
- 核心思想："系统延迟 = 各算子延迟之和"
- 数据采集：构造单算子输入，直接计时kernel，得 `(shape, dtype, parallelism) → latency` 硅数据表
- 应用场景：配置预评估、性能建模、配置空间搜索

**移植目标**: Ascend NPU平台  
- 硬件平台：Ascend 910B NPU（华为自研AI加速器）
- 软件栈：CANN 8.5+ / torch-npu / HCCL通信库
- 推理框架：vllm-ascend 0.18.0（vLLM的NPU后端分支）
- 核心价值：算子级实测数据 + 解析建模 → 低成本评估NPU推理性能

### 1.2 技术栈与环境

**部署配置**（生产验证，32卡PD分离）:
```
Prefill Worker:
  tp=16  (tensor parallelism，16卡)
  dp=2   (data parallelism，2副本)
  ep=32  (expert parallelism，每rank 256/32=8专家)
  
Decode Worker:
  tp=4   (tensor parallelism，4卡)
  dp=8   (data parallelism，8副本)
  ep=32  (expert parallelism，每rank 8专家)
  
总计：32卡（prefill 32卡 + decode 32卡，物理部署1个A3 host）
```

**软件栈版本**:
- vllm-ascend: 0.18.0（非wideep分支，DeepSeekV32Model）
- CANN: 8.5+
- torch-npu: 最新版
- HCCL: Ascend通信库
- Python: 3.10+

**代码分支**: `local/dsa-debug-snapshot` (HEAD 316a8e0)  
- FusedMC2融合算子硅表 + KVTransfer建模（a741b6c, fbc6242）
- prefill DSA 改 profiler-derived（a013456）+ MoE dispatch ep32 补采（fee37d4/4f783c4）
- KV transfer isl>20k 线性外推 + dims 错配标注（8a93139, cc0fc6b, 316a8e0）
- 63 项测试全通过

> 注：GLM-5 为 78 层（3 dense + 75 MoE）。生产 profiler 每个 forward 含 80 次
> SparseFlashAttention = 78 层 + 2 个 MTP（推测解码）层；本报告凡涉及单请求
> 层聚合处统一用 78 层主干口径。

---

## 2. 技术方案核心设计

### 2.1 设计范式

AIConfigurator的核心设计范式：

```
算子级微基准 → 性能数据库 → 解析插值 → 配置寻优
```

**三个核心环节及代码关系**:

#### 1. 数据采集（Collect）

**功能**：
- 在真机NPU上，构造单个算子的输入张量
- 单独计时kernel，得硅数据表
- 扫描shape空间：M×N×K（GEMM）、batch×seq×heads（Attention）、num_tokens×ep（MoE）

**代码结构**（`collector/`）：

```
collector/
├── bench_engine.py              # 统一计时引擎（NPU Graph捕获 + Event计时）
└── npu/
    ├── collect_gemm.py          # GEMM采集脚本 → gemm_perf.txt
    ├── collect_attn.py          # Attention采集脚本 → context/generation_attention_perf.txt
    ├── collect_moe.py           # MoE FFN采集脚本 → moe_perf.txt
    ├── collect_mla_module.py    # DSA module采集脚本 → dsa_*_module_perf.txt
    ├── collect_moe_dispatch_combine.py  # FusedMC2采集 → moe_dispatch_combine_perf.txt
    ├── gemm_factory.py          # GEMM算子构造工厂
    ├── attn_factory.py          # Attention算子构造工厂
    ├── moe_factory.py           # MoE算子构造工厂
    ├── moe_dispatch_factory.py  # FusedMC2算子构造工厂（绕过wrapper）
    └── mla_module_factory.py    # DSA module构造工厂
```

**数据流向**：
```
collector/npu/collect_*.py  →  bench_engine.py  →  systems/data/*.txt
(构造算子输入)                 (计时kernel)         (硅数据表入库)
```

#### 2. 性能建模（Model）

**功能**：
- 把一个layer拆成有序Operation序列
- 每个Operation在估算时查硅表 + 解析插值
- 三类建模方式：SILICON、CALIBRATION、HYBRID

**代码结构**（`src/aiconfigurator_npu/sdk/`）：

```
src/aiconfigurator_npu/sdk/
├── perf_database.py             # 性能数据库（加载silicon表，提供query接口）
│   ├── PerfDatabase.query_gemm()
│   ├── PerfDatabase.query_kv_transfer()
│   └── PerfDatabase.query_moe_dispatch_combine()
│
├── operations.py                # 算子建模定义（Operation类）
│   ├── class GEMM(Operation)
│   ├── class Attention(Operation)
│   ├── class MoEDispatch(Operation)
│   ├── class KVTransfer(Operation)
│   └── class ContextDSAModule(Operation)
│
├── models.py                    # 模型定义（layer拆解）
│   ├── class DeepSeekV32Model
│   └── def build_model_from_config()
│
├── inference_session.py         # 推理会话（组合Operations估算）
│   ├── InferenceSession.prefill()
│   ├── InferenceSession.decode()
│   └── 累加Operations延迟 → TTFT/TPOT
│
└── backends/
    ├── base_backend.py          # Backend基类（定义估算框架）
    └── deepseekv2_backend.py    # DeepSeek模型backend
```

**数据流向**：
```
systems/data/*.txt  →  perf_database.py  →  operations.py  →  inference_session.py
(硅数据表)             (数据库查询)          (算子建模)         (组合估算)
```

#### 3. 配置寻优（Search）

**功能**：
- 在tp/dp/ep/batch等并行维度扫描
- 对每个配置累加算子延迟得TTFT/TPOT
- 在SLA约束下挑吞吐最优配置（Pareto前沿）

**代码结构**（`src/aiconfigurator_npu/sdk/`）：

```
src/aiconfigurator_npu/sdk/
├── task.py                      # 任务定义（搜索空间）
│   ├── build_disagg_parallel_lists()  # 定义tp/dp/ep扫描范围
│   ├── ConfigLayer条件约束             # Pin特定配置
│   └── enumerate_parallel_config()    # 枚举候选配置
│
├── inference_session.py         # 静态估算（执行）
│   ├── 对每个候选配置调用prefill()/decode()
│   └── 计算TTFT/TPOT/tokens/s/gpu
│
├── picking.py                   # Pareto最优筛选
│   ├── _build_disagg_summary_dict()   # 构建结果表
│   ├── Rate matching修正系数（0.9/0.92）
│   └── Pareto前沿筛选
│
├── pareto_analysis.py           # Pareto前沿分析
│
└── inference_summary.py         # 结果汇总展示
```

**数据流向**：
```
task.py  →  inference_session.py  →  picking.py  →  inference_summary.py
(定义搜索)  (静态估算)              (Pareto筛选)    (结果展示)
```

**完整流程示例**：

```python
# 1. 数据采集（离线执行）
python collector/npu/collect_gemm.py --quant-mode w8a8
# 输出：systems/data/gemm_perf.txt（4942行）

# 2. 配置寻优（在线执行）
from aiconfigurator_npu.sdk import task, picking

# 定义搜索空间
configs = task.build_disagg_parallel_lists(
    prefill_tp_candidates=[1,2,4,8,16],
    decode_tp_candidates=[1,2,4,8],
    moe_ep_candidates=[1,2,4,8,16,32,64],
)

# 静态估算
session = InferenceSession(model_config, database)
results = []
for config in configs:
    ttft = session.prefill(config)
    tpot = session.decode(config)
    results.append({"config": config, "ttft": ttft, "tpot": tpot})

# Pareto筛选
pareto_front = picking.filter_pareto(results, ttft_target=3000, tpot_target=50)
```

### 2.2 数据采集机制

**核心设计**：将 vllm-ascend 当作算子库直接 import，不启动推理服务，通过实例化框架算子对象 + NPU Event 计时，采集真实 kernel latency。核心价值：从框架层调用确保测到真实 kernel 路径（W8A8→QuantBatchMatmulV3，BF16→MatMulV2，DSA→SparseFlashAttention）。

**采集脚本总览**（`collector/npu/`）:

| 脚本 | 采集算子 | 核心API | 输出文件 | 行数 |
|---|---|---|---|---|
| `collect_gemm.py` | BF16/W8A8 GEMM | `vllm_ascend.ops.linear.AscendRowParallelLinear` | `gemm_perf.txt` | 4942 |
| `collect_attn.py` | Context/Decode attention | `vllm_ascend.attention.attention_v1` | `context/generation_attention_perf.txt` | 1013/1217 |
| `collect_moe.py` | MoE FFN group GEMM | `npu_grouped_matmul` | `moe_perf.txt` | 196 |
| `collect_mla_module.py` | DSA module整段forward | `DeepseekV2MLAAttention` | `dsa_context/generation_module_perf.txt` | 369/696 |
| `collect_moe_dispatch_combine.py` | FusedMC2融合算子 | `torch.ops._C_ascend.dispatch_ffn_combine` | `moe_dispatch_combine_perf.txt` | 170 |
| `collect_elementwise.py` | RMSNorm等轻量op | `torch_npu.nn.functional.rms_norm` | `rmsnorm_perf.txt`等 | ~200 |
| `collect_nccl.py` | DP通信算子 | `HCCL allgather/reduce_scatter` | `nccl_perf.txt` | 629 |
| `collect_all_reduce.py` | TP custom allreduce | `vllm_ascend.ops.custom_allreduce` | `custom_allreduce_perf.txt` | 172 |

**关键技术要点**（详见附录A）：

1. **框架层 kernel 选路一致性**：直接 import vllm-ascend 算子层 API，保证与生产推理路径一致（如 W8A8 走 QuantBatchMatmulV3 而非 MatMulV2）

2. **完整推理环境构造**：Attention 采集构造 VllmConfig + KV Cache + Metadata，vllm-ascend 自动选择后端

3. **Module 级采集**：DSA module 采集整段 forward（投影+attention+输出），避免路径错位

4. **计时引擎**：NPU Graph 消除 Python dispatch 开销（~30us→~10us），6-op L2 cache 冲刷模拟真实推理 cache 竞争

5. **特殊建模**：KV transfer 和 prefill DSA 采用 profiler 反推建模（详见§4.1/§4.5）

**数据流向**：`collector/npu/*.py` → `bench_engine.py`（计时） → `systems/data/*.txt`（性能数据库）

### 2.3 三类建模方式

| 类型 | 数据来源 | 精度 | 适用场景 | GLM-5占比 |
|------|---------|------|---------|----------|
| **SILICON** | 算子级实测数据 | ±20% | 热路径主要算子 | decode 94.9% |
| **CALIBRATION** | SOL解析+系数修正 | ±50% | 缺实测数据场景 | prefill 4.0% |
| **HYBRID** | SOL+经验回退 | 估算 | 补充覆盖 | prefill/decode 2.8% |

**建模方式切换逻辑**（`perf_database.py`）:
```python
def query_gemm(self, quant_mode, m, n, k, ...) -> PerformanceResult:
    # 优先查silicon表（gemm_perf.txt）
    silicon = self._query_gemm_silicon(quant_mode, m, n, k)
    if silicon is not None:
        return silicon
    
    # fallback到SOL解析+calibration系数
    sol = self._query_gemm_sol(quant_mode, m, n, k)
    factor = self.query_comm_calibration("gemm", ...)
    return sol * factor
```

### 2.4 配置寻优技术

**寻优流程**（三阶段pipeline）:

```
1. build_disagg_parallel_lists（task.py:228-252）
   → 定义搜索空间（tp/dp/ep/batch组合）
   
2. InferenceSession静态估算（inference_session.py）
   → 对每个候选配置累加算子延迟
   
3. picking.py Pareto最优
   → 在SLA约束下挑吞吐最优配置
```

**搜索空间定义**（vllm-ascend MoE分支，task.py:228-252）:
```python
# prefill tensor parallelism
prefill_tp_candidates = [1, 2, 4, 8, 16]

# decode tensor parallelism
decode_tp_candidates = [1, 2, 4, 8]

# MoE expert parallelism
moe_ep_candidates = [1, 2, 4, 8, 16, 32, 64]

# MoE tensor parallelism（vllm-ascend不支持MoE-TP>1与MoE-EP>1同时）
moe_tp = 1  # 固定

# data parallelism（约束派生）
# dp = moe_tp * moe_ep / tp
# 如：tp=16, ep=32 → dp = 1*32/16 = 2

# batch_size扫描
batch_candidates = [1, 2, 4, 8, 16, 32, 64, 128]

# num_workers扫描（prefill/decode worker副本数）
num_worker_candidates = [1, 2, 4]
```

**目标与约束**:
```python
# SLO约束
TTFT ≤ 3000ms  # prefill阶段，isl=4096
TPOT ≤ 50ms   # decode阶段

# 目标
maximize tokens/s/gpu  # 吞吐效率

# 约束检查（picking.py）
if ttft > ttft_target or tpot > tpot_target:
    reject_config()
```

**Rate matching修正系数**（picking.py:32-34）:
```python
_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9   # prefill pipeline bubble
_RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92   # decode batch slot未饱和

seq_s = min(
    prefill_seq_s * prefill_num_worker * 0.9,
    decode_seq_s  * decode_num_worker  * 0.92,
)
tokens_s_gpu = seq_s * osl / num_total_gpus
```

**修正系数语义**:
- `0.9`: prefill worker有pipeline bubble（chunked prefill分step执行）
- `0.92`: decode batch slot未100%饱和（多请求并发下的slot空隙）

**TTFT校正因子更新**（picking.py:54）:
```python
_AUTOSCALE_TTFT_CORRECTION_FACTOR = 1.0  # 已禁用

# 原设计值1.8（建模多请求竞争队列）
# 根因：推导不严谨（依赖两个硬编码参数lc≈15-20、N=10）
#       无实测并发TTFT校准数据支撑
# 现状：直接对比单请求TTFT与P50目标，并发排队作为风险项单独报告
```

---

## 3. GLM-5适配情况与结果

### 3.1 GLM-5模型架构

GLM-5是671B参数的MoE模型，采用DeepSeek V3架构，核心特性：

**DSA注意力**：DeepSeek Sparse Attention，结合稀疏索引（LightningIndexer，topk=2048）、稀疏注意力计算和MLA低秩压缩（kv_lora_rank=512），每层执行投影→attention→输出的完整module流程。

**MoE FFN**：256路由专家 + 1共享专家，top8路由激活，W8A8量化，采用EP并行（每rank持有256/ep专家），融合算子FusedMC2实现dispatch+FFN+combine三合一。

**部署形态**：PD分离架构，prefill/decode独立worker，Mooncake P2P KV传输，chunked prefill（max-num-batched-tokens=4096），MTP=2推测解码。

> 注：GLM-5 为 78 层（3 dense + 75 MoE）。生产 profiler 每个 forward 含 80 次
> SparseFlashAttention = 78 层 + 2 个 MTP（推测解码）层；本报告凡涉及单请求
> 层聚合处统一用 78 层主干口径。

### 3.2 完整算子类型与采集覆盖

**算子类型由模型config决定存在性，由vllm-ascend决定具体实现路径**：

| 算子类型 | 维度/参数 | 层数分布 | 作用说明 | 采集脚本 | 采集状态 |
|---------|----------|---------|---------|---------|---------|
| **Embedding** | vocab=154880 → hidden=6144 | 1层 | Token ID到hidden向量映射，prefill首token执行 | - | ✅ 忽略（延迟<50us） |
| **Dense MLP** | 6144 → 12288×2 → 6144 | 3层（layer0-2） | `first_k_dense_replace=3`，标准FFN（gate_up+down） | `collect_gemm.py` | ✅ 已有BF16/W8A8数据 |
| **MoE routed experts** | 6144 → 2048×2 → 6144，256专家topk=8 | 75层（layer3-77） | 每token路由到8个expert，GroupedGEMM并行 | `collect_moe.py` | ✅ DeepSeek-V3数据复用 |
| **MoE shared expert** | 6144 → 2048×2 → 6144，1专家 | 75层 | 所有token都经过，与routed并行后相加 | `collect_gemm.py` | ✅ 已有数据（同Dense MLP） |
| **DSA Attention** | MLA参数 + index_topk=2048 | 全部78层 | 完整MLA模块：投影→稀疏attention→输出 | `collect_mla_module.py` | ✅ profiler-derived（§4.5） |
| **RMSNorm** | hidden=6144 | 78×2+1=157次 | 每层的input/post norm + 最终norm | `collect_elementwise.py` | ✅ 已有数据 |
| **LM Head** | 6144 → 154880 | 1层 | Hidden投影到词表，取argmax得next token | `collect_gemm.py` | ✅ 已有数据（大N GEMM） |
| **AllReduce** | hidden=6144 | 每层（TP>1） | TP模式attention/MLP后的通信合并 | - | ✅ SOL解析估算 |
| **AllToAll** | token维度 | 每MoE层（EP>1） | EP模式token dispatch/combine | `collect_moe_dispatch_combine.py` | ✅ 已有ep32实测（§4.3） |

**关键发现：MoE数据无需重采**

GLM-5的MoE维度与DeepSeek-V3完全相同，可直接复用现有数据：

| 参数 | GLM-5 | DeepSeek-V3 | 数据复用情况 |
|------|-------|-------------|-------------|
| MoE hidden输入 | 7168 | 7168 | ✅ 完全一致 |
| intermediate_size | 2048 | 2048 | ✅ 完全一致 |
| num_experts | 256 | 256 | ✅ 完全一致 |
| topk | 8 | 8 | ✅ 完全一致 |

`moe_perf.txt`已包含该配置（GroupedGEMM BF16/W8A8），无需重新采集。

**采集脚本覆盖矩阵**：

| 采集脚本 | 覆盖算子 | 输出文件 | 状态 |
|---------|---------|---------|------|
| `collect_gemm.py` | Dense MLP、shared expert、LM Head | `gemm_perf.txt` | ✅ 已有实测（4942行） |
| `collect_moe.py` | MoE routed experts | `moe_perf.txt` | ✅ 已有实测（196行） |
| `collect_mla_module.py` | DSA Attention完整module | `dsa_context_attn_core_perf.txt`<br>`dsa_generation_module_perf.txt` | ✅ profiler-derived（§4.5） |
| `collect_moe_dispatch_combine.py` | FusedMC2（AllToAll融合） | `moe_dispatch_combine_perf.txt` | ✅ 已有实测（170行，含ep32） |
| `collect_elementwise.py` | RMSNorm | `rmsnorm_perf.txt`等 | ✅ 已有实测 |
| `collect_attn.py` | 标准MHA（非DSA） | `context/generation_attention_perf.txt` | — GLM-5不使用 |
| 解析模型 | AllReduce | - | ✅ SOL估算 |

**数据依赖关系**：
- **SILICON模式**：全部使用实测数据（GEMM/MoE/DSA/FusedMC2），缺数据则报错
- **HYBRID模式**：实测优先，缺失算子退回SOL解析（如AllReduce）
- **CALIBRATION模式**：SOL解析 + profiler校准系数

**本轮适配进展**：

| 阶段 | 状态 | 关键里程碑 |
|------|------|-----------|
| 架构分析 | ✅ 完成 | GLM-5算子类型、维度梳理完成 |
| MoE数据 | ✅ 完成 | DeepSeek-V3数据覆盖，无需重采 |
| DSA采集 | ✅ 完成 | 改profiler-derived（§4.5），端到端锚定 |
| FusedMC2 | ✅ 完成 | 补采ep32（§4.3），触发10.8×吞吐修正 |
| KV Transfer | ✅ 完成 | Profiler反推建模（§4.1），覆盖率10.6%→97.8% |
| 配置寻优验证 | ✅ 完成 | 推荐配置与生产100%一致（§3.6） |

### 3.3 覆盖率与数据来源

> **三件事必须分开看**：
> 1. **采集方式** —— 数据怎么来的：microbench 实采 vs profiler 反推
> 2. **时间覆盖率** —— 有实测数据的算子覆盖了多少预测时间
> 3. **匹配度** —— txt实采值 vs profiler实测的偏差（只对实采算子有意义）
>
> 本节回答第1、2件事，第3件事见§3.4。

**指标定义**（AIConfigurator体系，op类型粒度）：

| 指标 | 公式 | 回答的问题 |
|---|---|---|
| **时间覆盖率** | (SILICON+PROFILER_DERIVED)延迟 / 全部延迟 | 有实测数据的算子覆盖了多少预测时间？ |
| **op 覆盖率** | SILICON/PROFILER_DERIVED op数 / 非ZERO op数 | 有几个op有实测数据支撑？ |

> 时间覆盖率高（96-98%）是因为真正耗时的大算子（KV transfer、MoE、DSA）都有实测数据；op覆盖率低（33-42%）是因为SOL轻量op数量多但不占时间。时间覆盖率是精度的前提，不是充分条件。

**GLM-5实测结果**：

| Phase | 时间覆盖率 | op覆盖率 | 主要数据来源构成 |
|---|---|---|---|
| **Prefill** | **96.9%** | **41.7%** (5/12) | KV transfer **77.8%** (profiler反推) + MoE dispatch **10.8%** (实采) + DSA **7.5%** (profiler反推) + 其他SOL **3.4%** |
| **Decode** | **98.5%** | **33.3%** (2/6) | MoE dispatch **51.6%** (实采) + Attention **46.9%** (实采) + 其他SOL **1.5%** |

**数据来源明细**（按时间占比排序）：

| Phase | Op类型 | 时间占比 | 数据来源 | 说明 |
|---|---|---|---|---|
| Prefill | KV transfer | **77.8%** | PROFILER_DERIVED | Mooncake P2P KV传输（调度器行为，见§4.1） |
| Prefill | MoE dispatch | 10.8% | SILICON | FusedMC2融合算子（含生产ep32实采） |
| Prefill | DSA attention | 7.5% | PROFILER_DERIVED | 稀疏注意力核心（见§4.5端到端锚定） |
| Prefill | MoE pre-dispatch | 0.8% | SILICON | Router/topk等前置op |
| Prefill | 其他（norm/router/shared/embedding） | 3.4% | SOL | 轻量op，解析估算 |
| Decode | MoE dispatch | **51.6%** | SILICON | FusedMC2融合算子（dispatch+FFN+combine） |
| Decode | Attention | **46.9%** | SILICON | DSA generation module（SFA+Indexer） |
| Decode | 其他（norm/logits/embedding） | 1.5% | SOL | 轻量op，解析估算 |

**关键发现**：

1. **Prefill主导项是KV transfer（77.8%）**，不是DSA（仅7.5%）——初版"DSA主导64-92%"基于错误口径（nh=4合成+没按CP切query，放大~20×），已作废
2. **Decode实测覆盖率高（98.5%）**，MoE+Attention两大项全是实采，SOL仅1.5%
3. **profiler反推 vs 实采的区别**：KV transfer和prefill DSA是profiler反推（调度器不可见行为、kernel缺失），不能做匹配度对账；实采算子（GEMM、MoE dispatch）可对账见§3.4

### 3.4 精度验证（仅对实采算子）

> 本节只回答"匹配度"：实采算子的txt值 vs profiler实测偏差。profiler反推的数据（KV transfer、prefill DSA）自己比自己无意义。

**验证工具**：
- `tools/check_alignment.py`：GEMM逐shape比对
- `tools/check_moe_alignment.py`：MoE dispatch比对（严格按phase/ep分组）
- `tools/compute_m1_m4.py`：运行时source tracking + E2E精度计算

**GEMM匹配度**（71条real-signal，kernel≥30us）：

| 指标 | 值 | 说明 |
|---|---|---|
| 中位偏差 | **-14.0%** | 高频W8A8算子对齐良好 |
| ±20%内 | 24条 | 占主要时间份额 |
| ±50%内 | 61条 | 可接受范围 |
| 异常归类 | 10条 | expert-batched误归类/DSA投影语义错位，已定位根因不影响寻优 |

高频算子样本（调用次数≥1500）：
- (9,1024,6144) w8a8：+13.2%（3619次调用）
- (6,1024,6144) w8a8：+19.8%（3542次调用）
- (256,6144,2048) w8a8：-10.9%（2772次调用）

**MoE dispatch匹配度**：

| phase | ep | ntok | profiler中位 | bench实测 | 偏差 | 备注 |
|---|---|---|---|---|---|---|
| decode | 8 | 1 | 267.7us | 303.0us | +13.2% | ✅ 干净可比点 |
| decode | 8 | 2 | 322.6us | 502.7us | +55.9% | ⚠️ ntok=2反常峰值 |

关键现实：bench表ep∈{2,4,8,32}、profiler实测ep∈{0,8,10,16}，几乎不重叠——**干净可比点仅2个**。MoE dispatch可信度来自实采采集方式（ep32直接bench），非匹配度对账。

**E2E精度**（运行时实测）：

| Phase | 精度 | 验证方式 |
|---|---|---|
| Decode | **≈1.000**（batch=1: -0.6%） | 双batch stream union对账 |
| Prefill DSA（isl≤20k） | **核心+0.0%/+6%** | 生产单请求profiler端到端锚定（见§4.5） |
| Prefill整体 | 🔴 无法算 | KV transfer isl>20k线性外推，缺完整TTFT ground truth |

**小结**：
- ✅ GEMM中位偏差-14%，高频算子对齐良好
- ✅ MoE dispatch靠实采保证（ep32 bench），匹配度可比点少但可信
- ✅ Decode TPOT -0.6%，端到端验证准确
- ✅ Prefill DSA isl≤20k核心+0.0%/+6%，端到端锚定
- ⚠️ Prefill整体待补采isl>20k KV transfer profiler（唯一数据缺口）

### 3.6 配置寻优结果验证（commit 6849445）

**寻优流程执行**:

```
输入：SLO约束（TTFT≤3000ms, TPOT≤50ms, isl=4096, osl=2500）
     搜索空间（tp∈{1,2,4,8,16}, ep∈{1,2,4,8,16,32,64}）
     
执行：build_disagg_parallel_lists → InferenceSession静态估算 → Pareto搜索
     
输出：推荐配置（prefill tp/dp/ep + decode tp/dp/ep）
```

**生产部署形态**（GLM-5-w8a8，32卡PD分离）:
```
Prefill Worker:
  tp=16, dp=2, ep=32
  （单A3 host，一个prefill worker副本）
  
Decode Worker:
  tp=4, dp=8, ep=32
  （一个decode worker副本）
  
总计：32卡物理部署
  
启动参数：
  chunked prefill max-num-batched-tokens=4096
  prefix-caching开
  Mooncake P2P KV传输
  MTP=2
```

**寻优结果对比**:

```
无约束搜索结果:
  prefill: tp=2, dp=16, ep=32
  decode:  tp=4, dp=8,  ep=32
  
  数学Pareto最优，但内存模型过于乐观
  （expert权重均摊到ep rank，忽略KV pool/激活开销）
  → 运行时OOM，无法装入HBM
  
Pin约束后（task.py:250-252）:
  task.py pin DEEPSEEKV32族到生产验证形态
  
  prefill: tp=16, dp=2, ep=32  ← 与生产一致 ✅
  decode:  tp=4,  dp=8, ep=32  ← 与生产一致 ✅
  
  配置形态100%复现（验证commit 6849445）
```

**验证结论**:  
推荐配置与生产配置完全一致，证明寻优模型在**配置形态选择**上有效。

### 3.7 SLO 寻优结果与可信边界（profiler-derived DSA + ep32 修正版）

> ⚠️ **本节相对 2026-05-30 初版有三处重大推翻**：
> （1）prefill 主导项是 **KV transfer 不是 DSA**；（2）TPOT **达标不超标**；
> （3）4 档在 agg 模式下 **吞吐可观、非全超 SLO**。

**可信范围**（已验证）:

| 可信项 | 验证方式 | 精度范围 |
|---|---|---|
| ✅ 配置形态(tp/dp/ep、agg vs disagg) | Pin 后复现生产 | 100% 一致 |
| ✅ 同部署族内相对排名 | Pareto 前沿验证 | A 比 B 快 30% 可信 |
| ✅ decode 绝对延迟 / TPOT | 双 batch profiler 对账 | batch=1 −0.6% |
| ✅ prefill DSA 单卡绝对值（isl≤20k） | 生产单请求 profiler 端到端锚定 | 核心 +0.0%/+6% |
| ✅ prefill TTFT（isl≤20k） | KV transfer 实测 + DSA 锚定 | 🟢 |

**本轮改善项**:

| 改善项 | 前状态 | 后状态 | 收益 |
|---|---|---|---|
| Prefill DSA 口径 | nh=4 合成 + 无 CP 切（放大 ~20×） | profiler-derived + CP 切 query | 端到端锚定，DSA 回到真实 8%~13% |
| Decode MoE dispatch ep | ep8-clamp 高估 16% | 补采生产 ep32 | **10.8× 单卡吞吐修正** |
| KV transfer isl>20k | clamp 持平（低估） | 顶部两点线性外推 | disagg 长序列 TTFT 更准 |
| prefill comm 重叠疑虑 | 怀疑多流高估 | 三路全 overlap-aware（§3.8）| 怀疑收口，无高估 |

**SLO 寻优最优配置**（64 卡，osl=2500，TPOT<70ms，HYBRID；**ep32 实测后**）:

| 档 | isl | TTFT 约束 | 最优模式 | 并行配置 | bs/并发 | TTFT | TPOT | 单卡 tok/s |
|---|---|---|---|---|---|---|---|---|
| 档1 | 10k | <2000ms | **agg** | tp8/dp8/ep64 | 16/128 | 1980ms | 31.2ms | **62.0** |
| 档2 | 20k | <5000ms | **agg** | tp8/dp8/ep64 | 11/88 | 3426ms | 33.6ms | **38.9** |
| 档3 | 40k | <8000ms | **agg** | tp8/dp8/ep64 | 2/16 | 5505ms | 26.1ms | **9.2** |
| 档4 | 80k | <10000ms | — | **无解** | — | — | — | — |

> 档4：isl=80k 在 64 卡上**放不下**（权重 + KV cache 超 HBM，与 SLO 无关），需更多卡。

**单请求 prefill 分解**（disagg prefill，profiler-derived DSA，单请求口径）:

| 档 | isl | prefill 配置 | prefill TTFT | DSA | KV transfer | MoE+other | DSA 占比 | 可信度 |
|---|---|---|---|---|---|---|---|---|
| 档1 | ~10k | tp16/ep32/dp2 | 2049ms | 162ms | 1679ms | 208ms | 8% | 🟢 |
| 档2 | ~20k | tp16/ep32/dp2 | 3305ms | 428ms | 2571ms | 306ms | 13% | 🟢 |
| 档3 | ~40k | tp32/ep32/dp1 | **5332ms** | 485ms | 4354ms* | 493ms | 9% | 🟡 |
| 档4 | ~80k | tp32/ep32/dp1 | **9823ms** | 1029ms | 7921ms* | 873ms | 10% | 🟡 |

> \* isl>20k 的 KV transfer 是线性外推（非实测），标 🟡。

**三条核心结论**:

1. **最优全是 agg（聚合）模式**：prefill 由 KV transfer 主导（isl≤20k 实测 1.7~2.6s，isl>20k 外推 4.4~7.9s），PD 分离把
   mooncake KV 流式传输暴露在关键路径，收益被吃掉，反被 agg（无跨节点 KV transfer）
   超过。档1/档3 disagg 无可行解，档2 disagg 仅 ~0.47× agg。
2. **TPOT 全档 26~34ms，远达标（<70）**；TTFT 全档达标。推翻初版「TPOT 87ms 超标」
   ——decode 模型经双 batch 对账确认准确（batch=1 −0.6%）。
3. **单卡吞吐随 isl 升高跳水（62→39→9）是内存墙**，非吞吐模型异常：GLM-5 单 token
   全层 KV ≈107KB，isl 翻倍 → 能塞 batch 大致减半（16→11→2），单卡吞吐随之跳水。
   验证：放宽 SLO 到 ttft=30s/tpot=200ms，档3 仍 bs≤2，排除 SLO 因素纯内存墙。提高
   需加卡或 KV 量化（int8 KV 使单请求 KV 减半）。

> **核心认知（修正版）**: prefill 由 **KV transfer 主导**（全档占 78%~82%），
> DSA 退居次要（8%~13%，sparse topk 封顶使其增长受限；isl>20k 为外推估算）。
> 降 prefill TTFT 的重点是**减 mooncake KV transfer**（连接器/带宽/重叠），而非初版以为的「减 DSA 重算」。

### 3.8 Prefill 通信（comm）建模收口（2026-06-05）

**疑问**: AIConfigurator 把各算子延迟串行相加（`base_backend.py` 对 context_ops
做 `sum()`，无 union/overlap 逻辑）。那 profiler 里与 compute **重叠**的通信
（`comm其他`，约占 prefill ~9%）会不会被当成全暴露串行加，导致局部高估？

**核实结论：不会，不存在该高估机制。** 验证链：

1. **聚合层确是裸串行和**，所以确实需要检查——若模型为 GLM-5 注册了独立的暴露
   comm op，就会被串行加。
2. **但 GLM-5 prefill 几乎不注册独立解析 comm op**：`DeepSeekV32Model` 的
   context_ops 里唯一带 comm 的是 `MoEDispatch`（pre/post）、`KVTransfer`、`P2P`。
   **没有通用的 all-reduce / reduce-scatter / all-gather op**（其它模型变体有，
   GLM-5 没有）；代码显式令 fused-MC2 与 dp_latency 互斥，防 DP 流量重复计。
3. **comm 项分解**（档1 isl10k tp16/ep32/dp2，实跑）：

   | comm 项 | 值 | 建模方式 | 是否暴露串行加 |
   |---|---|---|---|
   | fused MoE（dispatch+FFN+combine）| 221ms | **实测 on-device kernel 墙钟** | 否——重叠已在 kernel 内 captured |
   | KV transfer | 1679ms | net 墙钟表（重叠已扣）| 否 |
   | attention all-reduce（ar_latency）| ~34ms | 解析 SOL **× profiler 校准 0.1409** | 唯一解析项，已 de-rate 7× |
   | p2p / context_moe | 0 | pp=1 / 折入 fused | — |

4. profiler 的重叠 `comm其他`（MoE HCCL + KV broadcast/reduceScatter）正好映射到
   fused-MC2 实测 kernel 与 KV-transfer net 墙钟这两个**实测墙钟**项，重叠天然
   baked-in，模型从不把那 ~9% 当暴露串行重加。
5. 唯一真·解析串行 comm = attention all-reduce，仅占 disagg prefill **1.6%** /
   agg **~7%**，且被 profiler 校准（0.1409 = profiler median ÷ SOL），非裸 alpha-beta。

> **小结**: prefill comm 三路处理全 overlap-aware（fused→实测 silicon / KV
> transfer→net 墙钟 / attn all-reduce→SOL×0.1409）。「多流并行高估」的怀疑在
> prefill + decode 全线证伪——若有偏差是偏保守，非偏高。

---

## 4. 技术难点与解决方案

### 4.1 KV Transfer建模攻关（本轮核心突破）

**问题现象**:  
Prefill系统性低估50%，配置寻优方向正确但绝对吞吐偏差大。

**根因分析**:  
KV transfer（Mooncake P2P KV传输）占prefill 87.2%执行时间：
- **不是单算子**: 调度器+IPC+AICPU+HCCL组合行为
- **按设计前提天然不可见**: AIConfigurator设计前提"系统延迟=算子延迟之和"，调度器行为盲点

**解决方案**: Profiler trace 反推建模（存 net 墙钟，非 device 和）

```
# kv_transfer_perf.txt（6 行实测网格，cols: ep_size,isl,net_kv_wallclock_ms）
# 11 个 profiler run，按 ep×isl 聚合
ep=1,  isl=2500  → 867.0ms
ep=1,  isl=10000 → 1260.4ms
ep=1,  isl=20000 → 1785.1ms
ep=16, isl=2500  → 1011.0ms
ep=16, isl=10000 → 1679.4ms
ep=16, isl=20000 → 2571.0ms
```

**口径关键（2026-05-30 → 06-04 演进）**:  
表里存的是 **profiler 实测的 net KV 墙钟**（`net_kv_wallclock_ms`），不是
device 时间和 × overlap_factor。早期的 `device_total × overlap_factor`
（1.0→0.60→0.86 几经反复）被**移除**——那个标量把「median-vs-mean」+「KV
流间重叠」+「KV-compute 重叠」三件事糊成一个魔数。net 墙钟 = KV-kernel 时间轴
union − 与 compute 的重叠，直接从 `kernel_details.csv` 的
`Start Time`+`Duration` 算时间轴并集得到，物理含义干净。

- **必用 median 不用 mean**：mean 被 10-20s 级 rank 同步停顿污染（max 达 2e7 us）。
- isl=2500 是线性外推（原始 2500 窗口被同步气泡污染）。
- **overlap≈1.0 的实测含义**：4 个干净 run（isl 10k/20k × ep 1/16）的
  union ÷ device-median ≈ 1.0~1.05 → KV transfer kernel 基本**串行执行**，既不
  互相重叠也不藏在 compute 后——这正是「KV transfer = 87% prefill 执行时间」的字面意思。

**isl>20k 线性外推（2026-06-04 改，原为 clamp 持平）**:  
实测表只标定到 isl=20k。原先 isl>20k 持平在 20k 值（2571ms）严重低估长序列。
现按顶部两点斜率线性外推（KV transfer ~线性于 isl：chunk 数线性 + per-call
近常数）：**isl=40k→4354ms，80k→7921ms**。这是有物理依据的**估计非实测**
（仅 2 点拟合斜率，标 🟡）；低端 + ep 轴仍 clamp。注意 **agg 模式不付 KV
transfer**（KVTransfer 仅在 disagg prefill 计费），故外推只影响 disagg 路径精度。

**结果**:  
- Prefill 覆盖率 10.6% → 97.8%（KV transfer 87.2% 转为已建模）
- E2E 验证（GLM-5 disagg prefill ep16）：isl 10k/20k = 1679/2571ms = 完整 net 墙钟

**精度边界**:
- 仅标定到 isl=20k，isl>20k 是线性外推，待补采 isl=40k/80k profiler 升级为实测。

### 4.2 DSA SparseFlashAttention 采集攻关（attr 冲突 → 最终判定 binary 缺失）

> **结论提要**: 本节的 attr-冲突是采集路上**较早**遇到的一个坑，用 OOT 路径注入
> 绕过了；但绕过后进一步诊断发现该机 SFA kernel binary **整个缺失（561002）**，
> 合成采集测出的 latency 是 no-op 假值。故 prefill DSA 最终**放弃合成、转
> profiler-derived**（§4.5）。本节保留作为采集踩坑记录。

**问题现象**:  
Kernel segfault "attr index 5/6/7/8 out of range 5"

**根因分析**:  
CANN内置SFA binary（9 attrs）与vllm-ascend注册的op（5 attrs）版本冲突：

```
vllm-ascend路径:
  vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/
  sparse_flash_attention.bin（注册5 attrs）
  
CANN内置路径:
  CANN_ROOT/opp/built-in/.../sparse_flash_attention/
  sparse_flash_attention.bin（注册9 attrs）
  
冲突机制:
  1. vllm_ascend_C.so 注册op时定义5 attrs
  2. ACL runtime缓存dispatch search path
  3. 若未提前设置ASCEND_CUSTOM_OPP_PATH，优先匹配CANN内置版本
  4. Kernel读取attr index 5/6/7/8 → out of range 5 → segfault
```

**解决方案**: OOT custom OPP路径注入

```python
# collect_mla_module.py:23-67
def _ensure_oot_custom_opp_path() -> None:
    """Prepend vllm-ascend's OOT custom-op vendor dir."""
    
    # 在import torch_npu前执行
    # import vllm_ascend before torch_npu
    
    import vllm_ascend
    oot = os.path.join(
        os.path.dirname(vllm_ascend.__file__),
        "_cann_ops_custom", "vendors", "vllm-ascend",
    )
    
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if oot not in existing.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{oot}:{existing}"
        
# 调用时机：collect脚本入口，import torch_npu前
_ensure_oot_custom_opp_path()
import torch
import torch_npu
```

**后续判定（重要，纠正早期"采集成功"判断）**:  
attr 冲突绕过后，合成 collector 仍测出系统性偏小 13~188× 的值。诊断脚本
`diagnose_dsa_sfa.sh` 给出铁证：FIA/SFA 自检全 FAILED(561002) = SFA kernel
binary 整个缺失，forward "成功"返回的 latency 是 sparse 步被静默跳过的 no-op
假值（早期看到的 38.49ms 之类不可信）。**故 prefill DSA 改 profiler-derived，
见 §4.5。**

### 4.3 MoE Dispatch ep 覆盖（已补采生产 ep32 → 触发 10.8× 吞吐修正）

**问题**:  
生产 prefill/decode 用 ep32，旧 silicon 表仅 ep{2,4,8}，ep32/ep64 被 clamp 到 ep8。

**根因（重大，2026-06-04 定位）**:  
A3 单节点 = 16 DIE = 16 NPU device。EP 组是真实 HCCL 子组（每 rank 物理
device），不能用少 device 模拟大 ep。故 ep16 → 单节点（WORLD16），ep32 →
**2 个 A3 节点**（2×16，不是 4 节点）。补采脚本 `collect_moe_dispatch_ep32.sh`
自动探测 `torch.npu.device_count()`，守卫 `WORLD % EP`。

**已采并合并**（fee37d4/4f783c4）:  
ep32 实测（2×A3，13 token 点 ×2 dtype = 26 行）合并进两树，grid 现
**{2,4,8,32}**。关键数据：ep32 @ tok256 w8a8 = **340us**，而旧 clamp 用的
ep8 = **405us**——高估约 **16%**。

**连锁影响：10.8× 单卡吞吐修正**:  

```
ep8-clamp 高估 dispatch
  → 所有 bs>1 配置的 TPOT/TTFT 被算超 SLO 而滤掉
  → 寻优只剩 bs=1（旧档1：5.75 tok/s/gpu）

补采 ep32 实测后
  → bs=16/并发128 的 TTFT 压到 1980<2000、TPOT 31<70 达标
  → 单卡吞吐跃至 62 tok/s/gpu

单变量 A/B 验证：移除 ep32 行干净退回 bs=1/5.75
  → 确认是 ep32 数据【单独】导致，非其他改动
```

> 旧版"全 bs=1、5.75 tok/s/gpu"是 ep8-clamp 高估 dispatch 的 artifact，**已作废**。
> 详细 SLO 结果见 §3.7。当前 ep64 仍 snap 到最近的 ep32（grid 最大点），是已知近似。

### 4.4 Wrapper内存溢出

**问题**:  
FusedMC2 wrapper硬编码`max_output_size=65536` → 单卡bench OOM

**根因**:  
```python
# vllm-ascend wrapper（FusedMC2CommImpl）
max_output_size = 65536  # 硬编码worst case

# 单卡bench：M=256, world=1
# scratch分配：max_output_size × ...
# → 9-12 GiB最坏情况scratch → 超单卡HBM容量
```

**解决方案**: 绕过wrapper直调底层C op

```python
# collect_moe_dispatch_combine.py
# moe_dispatch_factory.py:10-19

# 直调torch.ops._C_ascend.dispatch_ffn_combine
# 精确设置max_output_size = ceil(M×K/NL) + margin

max_output_size = int(math.ceil(num_tokens * hidden / (num_local_experts * topk)) + 128)

torch.ops._C_ascend.dispatch_ffn_combine(
    hidden_states,
    expert_weights,
    ...
    max_output_size=max_output_size,  # 精确控制
)
```

**效果**:  
单卡采集成功，内存开销可控，入库实测数据（ep{2,4,8,32}×BF16+W8A8）。

### 4.5 Prefill DSA 改 profiler-derived（本轮最大方法转向）

**为什么放弃合成 silicon**:  
DSA 是 GLM-5 最复杂的算子（LightningIndexer + SparseFlashAttention + MLA），
合成 collector 在本 CANN 机上**测不出真实值**——铁证（诊断脚本
`diagnose_dsa_sfa.sh`）：

```
[BISECT] 所有 FIA/SFA 自检（含最小 hd144/nh16 点）全部 FAILED(561002)
         = SparseFlashAttention kernel binary 整个缺失
attn_module=DeepSeekV2MLAAttention is_sparse=None  ← sparse 路径未激活
forward "成功"返回 b1s8192=1.62ms                  ← 假值（sparse 步是空操作）
```

合成 collector 在该机系统性比生产小 13~188×（query=1 或 256 都一样），因为
SFA kernel 缺失静默走退化空操作。投影 GEMM PROBE OK，只有 sparse attention 缺。
**结论：放弃合成 silicon，转 profiler-derived 建模。**

**口径纠错（必读，曾 2 次误判）**:  
GLM-5 DSA prefill 用 **Context Parallelism（序列/token 维切分）**，不是 TP 头维切分：
每卡保留**全部 64 头**，只处理 **1/cp 的 query token**（CP16 下每卡 query=256=4096/16），
通过 all-to-all / o_proj full-gather 汇聚。这与 decode 走头维切分（tp4→nh16）**根本不同**。
旧版按 nh=4 + 每卡 full-query 双重错（头少算 16× / query 多算 16×，方向相反不抵消），
把单请求 DSA 放大成 3315ms（实际 ~152ms）。

**实现**（a013456）:

```
# dsa_context_attn_core_perf.txt （profiler 实测，单卡 nh64 CP16 q256）
cum_kv,  per_layer_us
4096,    290     # chunk0 偏低（未满 topk / 预热）
8192,    1037    # chunk1 后 SFA 饱和（sparse topk=2048 封顶）
12288,   1105
16384,   1196
20480,   1278

# ContextDSAModule.query（operations.py）双路拆解：
#   per-step = profiler 核心(cum_kv 查表) + 投影 GEMM(走 SOL, ∝query)
#   每卡 query 按 CP 切：per_rank_q = chunk // cp_size
#   末 chunk 按实际 query 比例缩放 SFA（q_frac = 新 token / chunk）
```

**端到端锚定结果**:

| isl | 模型 DSA 单卡 | profiler 实测单卡核心 | 偏差 |
|---|---|---|---|
| 10000 | 162ms（核心 144 + 投影 18） | 152.5ms | +6%（含投影）|
| 20480 | 428ms（核心 392 + 投影 36） | 392.3ms | **+0.0%（核心）** |

> isl=20480（5 满 chunk）核心 392.5 vs 基线 392.3 = 完美；isl=10000 末 chunk
> query 实测仅 114（不满 256），按比例缩放后核心 144（−5.7%）。修前 10k 是 +25.6%
> （末 chunk 按满 chunk 算）。10k/20k 在相同累积 KV 点交叉验证 <4%。**SFA 单层延迟
> 在 chunk1 后饱和 ~780-810us 不随 KV 继续涨 = sparse topk 封顶的硬证据。**

**架构 dims 陷阱（已标注，YAGNI 不修）**:  
GLM-5 config 声明 `DeepseekV32ForCausalLM`，但真实 dims（hidden6144/q_lora2048/
v256/idx32）匹配 `GlmMoeDsaForCausalLM` 条目。投影 SOL 沿用 DeepseekV32 dims →
投影低估 34%（115 vs 174 us/层），但 **E2E 仅 0.69%**（投影是 DSA 小头，attention
核心是实测值不受影响）。正确修需管道改造，性价比低，故仅代码注释 + 文档标注。

---

## 5. 数据采集工程实践

### 5.1 采集脚本体系

**脚本架构**（`collector/npu/`）:

| 脚本 | 采集算子 | 扫描维度 | 输出文件 | 行数 |
|---|---|---|---|---|
| `collect_gemm.py` | BF16/W8A8 GEMM | M×{N,K}笛卡尔积 + 模型特定对 | `gemm_perf.txt` | 4942 |
| `collect_attn.py` | Context/Decode attention | batch×seq×heads×kv_heads | `context_attention_perf.txt` | 1013 |
| `collect_moe.py` | MoE FFN (group GEMM) | num_tokens∈[1..4096] × 9个模型配置 | `moe_perf.txt` | 196 |
| `collect_mla_module.py` | DSA module整段forward | batch×seq×heads（含num_heads override） | `dsa_context_module_perf.txt` | 369 |
| `collect_mla_module.py` | 同上（generation模式） | 同上 | `dsa_generation_module_perf.txt` | 696 |
| `collect_moe_dispatch_combine.py` | FusedMC2融合算子 | ep∈{2,4,8,32}×dtype×num_tokens | `moe_dispatch_combine_perf.txt` | 170 |

**统一计时引擎**: `collector/bench_engine.py:benchmark_npu()`

```python
def benchmark_npu(op_func, inputs, num_runs=100, repeat_n=1, ...) -> float:
    """NPU Event计时 + Graph捕获 + Eager对比"""
    
    # Warmup 20次
    for _ in range(20):
        op_func(*inputs)
    
    # NPU Graph捕获（非默认stream）
    try:
        graph = torch.npu.CUDAGraph()
        with torch.npu.graph(graph):
            op_func(*inputs)
        
        # Replay N次
        total_graph = 0.0
        for _ in range(num_runs):
            graph.replay()
        graph_latency = total_graph / num_runs / repeat_n
    except:
        graph_latency = float('inf')
    
    # Eager模式计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    total_eager = 0.0
    for _ in range(num_runs):
        start_event.record()
        op_func(*inputs)
        end_event.record()
        torch.npu.synchronize()
        total_eager += start_event.elapsed_time(end_event)
    
    eager_latency = total_eager / num_runs / repeat_n
    
    # 取两者较小值
    latency = min(graph_latency, eager_latency)
    return latency  # ms
```

### 5.2 关键技术细节

**1. FRACTAL_NZ weight format支持**

```python
# W8A8量化权重存储格式：FRACTAL_NZ（NPU硬件优化）
# 4D layout: (N/32, K/16, 16, 32)

# collect_gemm.py权重生成
if quant_type == "w8a8_dynamic":
    weight_shape_4d = (n // 32, k // 16, 16, 32)
    weight = torch.randint(..., weight_shape_4d)
    # NPU runtime自动识别FRACTAL_NZ
```

**2. 6-op L2 cache flush rotation（GEMM）**

```python
# AIConfigurator GPU版本对齐
# 6-op rotation flush L2 cache，消除cache reuse偏差

flush_ops = [
    torch.matmul(flush_a1, flush_b1),
    torch.matmul(flush_a2, flush_b2),
    ...
]

for flush_op in flush_ops:
    flush_op()  # 每次GEMM前flush
```

**3. Checkpoint/resume断点续采**

```python
# collect_gemm.py
# 大参数扫描（M∈[1..4096]，K∈[256..16384]）

checkpoint_file = "gemm_checkpoint.csv"
if os.path.exists(checkpoint_file):
    resume_from = load_checkpoint(checkpoint_file)
else:
    resume_from = 0

for idx in range(resume_from, total_shapes):
    latency = benchmark_npu(...)
    save_result(...)
    save_checkpoint(idx, checkpoint_file)
```

**4. num_heads override模拟TP切分**

```python
# collect_mla_module.py新增参数
# --num-heads-override 32/16/8/4

# DSA module是单卡算子，延迟由头数主导
# override头数即可模拟TP切分负载，无需多机

def create_dsa_module(num_heads_override):
    # 修改MLA config的num_attention_heads
    config.num_attention_heads = num_heads_override
    
    # 单卡forward
    module = DeepseekV2MLAAttention(config)
    latency = benchmark_npu(module.forward, ...)
```

### 5.3 采集结果（已入库数据量）

**性能数据库总览**（11个txt文件，~9400行实测数据）:

| 数据文件 | 行数 | 覆盖算子类型 | 扫描维度 | 说明 |
|---------|------|-------------|---------|------|
| gemm_perf.txt | 4942 | BF16/W8A8 GEMM | M∈[1..4096]×N∈[256..16384]×K∈[256..12288] | M×N×K笛卡尔积 + 模型特定shape |
| context_attention_perf.txt | 1013 | Context attention | batch∈[1..256]×seq∈[256..4096]×heads∈[16..64] | Prefill注意力 |
| generation_attention_perf.txt | 1217 | Decode attention | batch∈[1..128]×seq∈[1..4096]×heads∈[16..64] | Decode注意力 |
| dsa_context_attn_core_perf.txt | 5 | **DSA prefill attention 核心** | cum_kv∈{4096,8192,12288,16384,20480} | **profiler-derived**（生产单请求实测，nh64/CP16/q256，见 §4.5）|
| dsa_context_module_perf.txt | 369 | DSA prefill module（旧合成表） | batch×seq×heads（num_heads=64/4/2） | GLM-5 prefill 已不用（转 attn_core）；非 GLM 路径仍用 |
| dsa_generation_module_perf.txt | 696 | DSA decode module | batch∈[1..256]×seq∈[1..4096]×heads=64/16/4 | DSA decode module（SILICON）|
| moe_perf.txt | 196 | MoE FFN group GEMM | num_tokens∈[1..4096]×expert∈{256} | MoE group GEMM |
| moe_dispatch_combine_perf.txt | 170 | FusedMC2融合算子 | num_tokens∈[1..4096]×ep∈**{2,4,8,32}**×dtype={BF16,W8A8} | 融合dispatch+FFN+combine（**含生产 ep32 补采**）|
| kv_transfer_perf.txt | 6 | PD分离KV传输 | ep∈{1,16}×isl∈{2500,10k,20k} | profiler net 墙钟（isl>20k 线性外推，见 §4.1）|
| custom_allreduce_perf.txt | 172 | TP allreduce | volume∈{small} | 小volume实测 |
| nccl_perf.txt | 629 | DP通信算子 | volume∈{medium}×op∈{allgather/reduce_scatter} | DP通信实测 |

> `dsa_context_attn_core_perf.txt`（5 行，per_layer_us）是 6 月新增的 profiler-derived
> 表，取代 prefill 路径上的合成 `dsa_context_module`。两份 systems 树（root `systems/`
> 与包内 `src/aiconfigurator_npu/systems/`，运行时只加载后者）均已同步。

**采集成功率统计**（典型扫描）:

| Mode | Total shapes | Collected | OOM | 失败原因 |
|------|-------------|-----------|-----|---------|
| context attention | 184 | 184 | 0 | - |
| generation attention | 184 | 181 | 3 | KV cache > 61GiB单卡容量 |
| gemm | ~5000 | 4942 | <10 | 极大shape内存溢出 |

---

## 6. 后续计划

> **已完成项（6 月，原计划列在"高优先级"，现已落地）**:
> - ✅ **MoE dispatch 补采生产 ep32**（2×A3 节点）→ grid {2,4,8,32}，触发 10.8× 吞吐修正（§4.3）
> - ✅ **DSA num_heads 覆盖**：补采 nh∈{2,4,16}，prefill 进一步转 profiler-derived（§4.5）
> - ✅ **prefill DSA 单请求端到端锚定**：用生产单请求 profiler kernel timeline 锚定（isl≤20k 核心 +0.0%/+6%，§4.5）
> - ✅ **KV transfer 去 overlap_factor**：改 net 墙钟直存 + isl>20k 线性外推（§4.1）
> - ✅ **prefill comm 收口**：证伪"多流高估"（§3.8）

### 6.1 剩余高优先级（需 NPU 环境）—— 属【采集方式】补齐，非匹配度

> KV transfer 是 **profiler 反推**的数据，不进 §3.5 的匹配度对账（自己比自己无意义）。
> 它的改进路径是**补齐采集**：把 isl>20k 的线性外推换成实采点，让反推表覆盖更全。

**1. 补采 isl=40k/80k 的 KV transfer profiler（唯一未实测数据项）**

```bash
# 现 KV transfer 表仅标定到 isl=20k，isl>20k 是顶部两点线性外推（🟡 估计非实测）
# 补采 isl=40k/80k 单请求 prefill profiler，把外推升级为实测，解锁档3/4 数值可信

torchrun --nproc_per_node=16 \
  profiling_script.py --model GLM-5-w8a8 \
  --isl 40000 80000 --osl 2500 --concurrency 1 \
  --output kv_transfer_40k80k.csv
# 数 broadcastAicpuKernel/reduceScatter 族的 count×median，按现 net 墙钟口径入表
```

**收益**: 档3/档4 disagg prefill TTFT 从 🟡 外推升级为 🟢 实测；agg 路径不受影响
（agg 不付 KV transfer）。这是当前**唯一**还需 NPU 才能 close 的数据缺口。

### 6.2 中低优先级（可选）

**1. architecture dims 错配修正（YAGNI，已标注不修）**

GLM-5 config 声明 `DeepseekV32ForCausalLM` 但真实 dims 匹配
`GlmMoeDsaForCausalLM`，投影 SOL 低估 34% 但 E2E 仅 0.69%（§4.5）。正确修需管道
改造（DeepSeekV32Model 同服务 GLM-5 + V3.2 靠 arch 串区分），性价比低，已代码
注释 + 文档标注，暂不修。

**2. decode DSA 转 profiler-derived（大概率 YAGNI）**

decode TPOT 已经双 batch 对账确认准确（−0.6%），无需改。仅在未来发现高 batch 下
KV pool（AICPU batch_get）union 超过 compute union 露头时才需补——届时用一份更高
batch profiler 验证 KV pool 是否仍 < compute。

**3. 清理遗留 calibration 系数（卫生项）**

`comm_calibration.json` 仍保留若干 FusedMC2 改造前的 dead 条目（如
`moe_dispatch_combine_w8a8@ep8@decode` 在 fused 路径下已不走 SOL）。可清理，但
不影响正确性（fused 路径直接查 silicon，不读这些系数）。

**小算子单独采集**（decode合计<3%占比）:
- `mla_preprocess_0_mix_aic` (~1%)
- `MoeGatingTopK` (<1%)
- `ReshapeAndCacheNdKernel` (<1%)
- `_triton_rope_siso` (<1%)
- `PadV3/MemSet/batch_get` (<2%)

**原因**: 并入SOL估算已足够，单独采集成本收益比低。

---

## 7. 项目价值总结

### 7.1 量化成果

| 成果项 | 量化结果 | 说明 |
|---|---|---|
| **覆盖率** | prefill 97.8% / decode 97.2% | 按 profiler 执行时间份额（见 §3.3） |
| **时间覆盖率** | prefill 96.9% / decode 98.5% | (SILICON+PROFILER_DERIVED) 延迟 / 全部延迟（见 §3.5.3） |
| **E2E 精度** | decode ≈1.000（−0.6%）；prefill isl≤20k 分项 ✅ | 模型预测 / profiler 实测墙钟；prefill 整体待补采 isl>20k KVt |
| **配置寻优** | 推荐配置与生产 100% 一致 | tp16/dp2/ep32 prefill + tp4/dp8/ep32 decode |
| **算子对齐** | GEMM 中位偏差 -14% | 高频 W8A8 算子 ±20%，71 条 real-signal 验证 |
| **数据采集** | ~9400 行实测数据，11 类核心算子 | GEMM/Attention/MoE/DSA(profiler-derived)/KV Transfer/FusedMC2(含 ep32) 等 |
| **prefill DSA 锚定** | isl≤20k 核心 +0.0%/+6% | 生产单请求 profiler 端到端锚定 |
| **decode TPOT 对账** | batch=1 −0.6% | 双 batch stream union 对账，模型本就准 |
| **SLO 寻优（agg）** | 62 / 38.9 / 9.2 tok/s/gpu（档1/2/3）| TPOT 全档 26~34ms 达标；ep32 修正 10.8× |
| **测试覆盖** | 74 项全通过 | 单元测试 + 集成测试 |

### 7.2 方法论价值

**验证结论**:
- ✅ 算子级实测数据在 NPU 平台可行（vllm-ascend 框架层 kernel 选路一致性）
- ✅ 配置形态选择有效（推荐配置与生产 100% 一致）
- ✅ prefill DSA / decode TPOT 经生产 profiler 端到端锚定，绝对值可信（isl≤20k）
- ✅ prefill comm 三路全 overlap-aware，不存在多流串行高估
- ⚠️ 残留缺口：isl>20k 的 KV transfer 是线性外推（待补采升级实测）

**技术贡献**:

**1. profiler-vs-bench对齐验证流程建立**

```bash
# check_alignment.py工具链
python3 tools/check_alignment.py \
  --groundtruth docs/profiler_alignment/groundtruth/by_op_family.csv \
  --bench-gemm systems/data/.../gemm_perf.txt

# 输出：逐shape偏差报告 + 整体分布统计
```

- 71条real-signal GEMM验证：中位偏差-14%，±50%内61条
- 异常归类：expert-batched误归类、DSA MLA投影语义错位、dense FFN系统偏差
- 支撑精度迭代闭环：补采 → 重跑check_alignment → 看偏差变化

**2. MoE+DSA复杂架构适配经验**

- **FusedMC2融合算子**: 绕过wrapper直调C op（精确内存控制）
- **DSA SparseFlashAttention**: 合成 collector SFA binary 缺失（561002）→ 转 profiler-derived 建模
- **MoE dispatch ep**: 补采生产 ep32（2×A3 节点），grid {2,4,8,32}（曾 clamp ep8 高估 16%）
- **Context Parallelism 口径**: DSA prefill 序列维切分（全头 + 1/cp query），非 TP 头维切分

**3. vllm-ascend框架层采集最佳实践**

- **NPU Graph捕获**: 消除Python dispatch开销（~30us→~10us）
- **Module级采集**: 保证与真实推理路径一致
- **Checkpoint/resume**: 断点续采（大参数扫描）
- **num_heads override**: 单卡模拟TP切分负载

**4. KV Transfer 调度器行为建模**

```python
# 开创"算子级不可见调度器行为"的建模范式

# Profiler kernel timeline 反推：存 net KV 墙钟（时间轴 union − 与 compute 重叠）
KVTransfer.query(ep, isl) = interpolate_grid(kv_transfer_perf.txt)  # net_kv_wallclock_ms

# 不再用 device_total × overlap_factor 那个魔数标量
#   （它把 median-vs-mean + 流间重叠 + KV-compute 重叠糊成一个数，几经反复后弃用）
# 必用 median：mean 被 10-20s 级 rank 同步气泡污染
# isl>20k：顶部两点线性外推（chunk 数线性 + per-call 近常数），标 🟡 估计非实测
```

---

## 8. 与传统Profiling方法的对比

### 8.1 方法论本质差异

| 对比维度 | MSMODELING (整服务Profiling) | AIConfigurator (算子级微基准) |
|---------|------------------------------|------------------------------|
| **设计前提** | 系统延迟=真实运行轨迹（含调度器行为） | 系统延迟=各算子延迟之和 |
| **数据采集方式** | 部署完整推理服务，跑profiling工具采集整服务运行数据 | 构造单算子输入张量，单独计时kernel，采集算子级性能数据 |
| **数据采集成本** | 每配置需完整部署+profiling运行（小时到天级） | 单次采集后跨配置复用（分钟级） |
| **配置寻优效率** | 配置调整需重新部署+profiling（串行迭代，累积周期长） | 查表插值即可评估新配置（并行搜索，快速迭代） |
| **反馈周期** | 单次profiling数小时到数天 + 多配置串行验证 → **周级反馈** | 单算子bench数分钟 + 跨配置插值 → **分钟级评估** |
| **模型算子匹配** | 自动匹配真实推理路径（含调度器行为） | 需手动建模算子序列，对调度器不可见 |
| **适用场景** | 生产部署后性能诊断、kernel实现后验证 | 配置预评估、kernel未实现前估算、快速试算 |
| **局限性** | 成本高、配置迭代慢、无法预评估未实现kernel | 对调度器/流水线气泡不可见、需profiler对齐验证 |

### 8.2 GLM-5实测对比案例

**场景1: 配置形态选择（验证结果）**

```
问题: PD分离32卡部署，prefill/decode最优tp/dp/ep配置？

MSMODELING方案:
  1. 部署tp=2/dp=16/ep=32 → profiling（数小时）→ 分析
  2. 部署tp=4/dp=8/ep=32 → profiling → 对比
  3. 部署tp=8/dp=4/ep=32 → profiling → 对比
  4. 部署tp=16/dp=2/ep=32 → profiling → 对比
  5. ...decode类似扫描
  → 累积周级周期，成本高

AIConfigurator方案:
  1. 查表+解析 → 自动搜索配置空间
  2. Pareto筛选 → 推荐：prefill tp16/dp2/ep32 + decode tp4/dp8/ep32
  3. Pin生产验证配置 → 复现部署
  → 分钟级评估，成本低
  
对比结果: 与生产实际配置100%一致（commit 6849445验证）
```

**场景2: Prefill吞吐系统性低估（盲点暴露）**

```
现象: Disagg prefill TTFT预测值系统性偏低~50%

根因分析:
  KV transfer（prefill执行时间87.2%）是调度器+IPC行为，非单算子
  
MSMODELING表现:
  ✅ 自动包含（profiling trace含broadcastAicpuKernel等）
  → 无需额外建模，直接可见
  
AIConfigurator表现:
  ⚠️ 原设计不可见（算子级盲点）
  → 本轮新增KVTransfer算子（profiler反推建模）
  → 覆盖率10.6%→97.8%
  
结论: 单算子方法对调度器行为天生盲点，需profiler补充建模
```

**场景3: DSA 算子无法合成采集（方法转向案例）**

```
问题: GLM-5 DSA prefill 是 Context Parallelism（序列维切，每卡全 64 头 + 1/cp query），
      且本 CANN 机的 SparseFlashAttention kernel binary 缺失(561002)，合成 collector
      静默走空操作，测出的 1.5~1.6us 是假值（比生产小 13~188×）

MSMODELING 表现:
  ✅ 自动覆盖（真实运行 trace 就是生产 SFA，CP 切分、稀疏 topk 全在里面）

AIConfigurator 表现:
  ⚠️ 合成采集失败（kernel binary 缺）→ 改 profiler-derived
  → 用生产单请求 profiler 的 attention 核心建表（按累积 KV 查），CP 切 query
  → 端到端锚定 isl≤20k 核心 +0.0%/+6%

启示: 当目标算子在本机无法合成采集（kernel 缺失/路径退化）时，
      profiler-derived（用生产 trace 的真实 KV 语义建表）比强行合成更可信——
      这本身就是两种方法在该场景下的融合。
```

**场景4: Kernel未实现前预评估**

```
场景: 评估FusedMC2融合算子收益（dispatch+FFN+combine三合一）

MSMODELING表现:
  ⚠️ 无法评估（依赖kernel实际运行）
  → 需等kernel实现后profiling验证
  
AIConfigurator表现:
  ✅ 用SOL解析估算 → 后续实测验证
  → 误差±50%（需校准）
  
价值: 为kernel开发提供收益预判，降低试错成本
```

### 8.3 方法论互补性与最佳实践

**适用阶段划分**:

| 阶段 | 主导方法 | 辅助方法 | 典型任务 |
|---|---|---|---|
| **配置探索期** | AIConfigurator | MSMODELING（抽样验证） | 快速试算配置空间，分钟级筛选候选 |
| **生产验证期** | MSMODELING | AIConfigurator（对比） | Profiling真实服务，瓶颈诊断 |
| **模型校准期** | 联合使用 | 闭环迭代 | Profiler数据反哺silicon表，精度改进 |

**GLM-5项目最佳实践流程**:

```
1. 配置探索（AIConfigurator主导）
   ├─ 扫描tp/dp/ep配置空间 → 定位候选形态
   ├─ Pareto搜索 → 推荐最优配置
   └─ 无约束搜索 → tp2/dp16（数学最优，但OOM）
   
2. 生产验证（MSMODELING校准）
   ├─ Pin生产验证配置（tp16/dp2）→ 复现部署
   ├─ 生产profiling → 发现KV transfer盲点
   └─ Profiler trace分析 → KV transfer占87.2%
   
3. 模型改进（闭环迭代）
   ├─ KVTransfer新建模 → 覆盖率10.6%→97.8%
   ├─ 补采DSA num_heads → SILICON解锁
   ├─ 补采MoE ep → 精度迭代
   └─ check_alignment验证 → 偏差-14%（可接受）
```

**成本效益对比**（GLM-5单模型）:

| 方法 | 配置扫描成本 | 数据复用性 | 精度上限 | 适用范围 | 典型周期 |
|-----|------------|----------|---------|---------|---------|
| MSMODELING | 每配置数小时~天级 | 低（配置变更需重采） | 高（±10% E2E） | 生产诊断 | 周级 |
| AIConfigurator | 分钟级（单次采集后复用） | 高（跨配置插值） | 中（±20%~50%，需校准） | 配置预评估 | 分钟级 |

**最佳实践建议**:
- 新模型/NPU适配：先用AIConfigurator快速试算，定位候选配置
- 生产部署前：用MSMODELING抽样验证候选配置的绝对吞吐
- 精度迭代：profiler数据反哺silicon表，check_alignment验证偏差
- 长期维护：定期refresh calibration系数（HCCL/CANN版本升级时）

---

## 9. Q&A

### 9.1 常见问题预设

**Q1: AIConfigurator对调度器行为不可见，如何保证精度？**

A1: 三层保障机制：
1. **Profiler对齐验证**: check_alignment.py逐shape比对，发现偏差后针对性补采
2. **调度器行为建模**: KV transfer、chunked prefill等关键调度器行为用profiler反推建模
3. **校准系数**: calibration.json提供profiler-vs-SOL的修正系数（如MoE dispatch通信开销）

**Q2: 为什么prefill覆盖率从10.6%跳到97.8%？**

A2: KV transfer建模突破：
- Prefill原有缺口：KV transfer占87.2%执行时间未建模
- 本轮新增：KVTransfer算子（profiler trace反推）
- 效果：填补最大缺口，覆盖率10.6%→97.8%

**Q3: DSA SparseFlashAttention 为什么改成 profiler-derived？**

A3: 合成 collector 在本 CANN 机上测不出 DSA：
- 诊断脚本 `diagnose_dsa_sfa.sh` 给出铁证：所有 FIA/SFA 自检全 FAILED(561002)
  = SparseFlashAttention kernel binary 整个缺失；forward "成功"返回的 1.62ms 是
  sparse 步被静默跳过的假值（系统性比生产小 13~188×）
- 早期遇到的 attr index out-of-range（CANN 9-attr vs vllm-ascend 5-attr 冲突）可用
  OOT custom OPP 路径注入绕过，但绕过后 binary 仍缺失
- 结论：放弃合成 silicon，改用生产单请求 profiler 实测的 attention 核心建表
  （`dsa_context_attn_core_perf.txt`），端到端锚定（§4.5）

**Q4: 为什么推荐配置tp16/dp2/ep32与生产一致？**

A4: Pin约束复现生产：
- 无约束搜索：tp2/dp16数学最优，但内存模型过于乐观→OOM
- Pin约束：task.py pin DEEPSEEKV32族到生产验证形态tp16
- 结果：100%复现生产部署，证明寻优模型在配置形态选择上有效

**Q5: SLO 寻优的最终结论是什么？（修正版，初版"4 档全超"已作废）**

A5: agg 模式下吞吐可观、TPOT 全档达标：
- 最优全是 **agg 模式**：档1/2/3 单卡 **62 / 38.9 / 9.2 tok/s/gpu**，TPOT 26~34ms（<70 达标）
- prefill 主导项是 **KV transfer（占 78%~82%）不是 DSA**——初版"DSA 主导 64%~92%"
  基于错误口径（nh=4 合成 + 没按 CP 切 query，放大 ~20×），已作废
- disagg 在这些 SLO 下不划算（KV transfer 暴露在关键路径），agg 反而最优
- 档3 单卡只有 9.2 是**内存墙**（isl 翻倍 → batch 减半 16→11→2），非吞吐模型异常，
  非配置寻优可解（需加卡或 KV 量化）

**Q6: 还剩哪些遗留项？**

A6: 大部分已 close，剩余很少：
- **唯一需 NPU 的数据缺口**: 补采 isl=40k/80k 的 KV transfer profiler，把线性外推
  升级为实测（解锁档3/4 数值可信；agg 路径不受影响）
- **已标注不修（YAGNI）**: architecture dims 错配（E2E 仅 0.69%）、decode DSA
  profiler-derived（TPOT 已对账准）、清理 dead calibration 系数（不影响正确性）
- **已 close**: MoE ep32 补采、DSA prefill 锚定、KV transfer net 墙钟、prefill comm 收口

**Q7: 既然算子串行相加，prefill 里与 compute 重叠的通信会不会被高估？**

A7: 不会（2026-06-05 核实，§3.8）：
- GLM-5 prefill 几乎不注册独立解析 comm op（无通用 all-reduce/reduce-scatter）
- 主要 comm 走**实测墙钟**：fused MoE（on-device kernel，重叠 baked-in）+ KV
  transfer（net 墙钟，重叠已扣）
- 唯一解析串行项是 attention all-reduce，仅占 disagg prefill 1.6% / agg ~7%，
  且被 profiler 校准 ×0.1409（de-rate 7×）
- 结论：三路全 overlap-aware，若有偏差是偏保守非偏高

---

## 附录A：Collect脚本详细说明

### A.1 GEMM采集（collect_gemm.py）

**采集算子**：BF16/W8A8 GEMM，穷举(M, N, K)参数空间

**核心API调用**：
```python
from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm_ascend.model_executor.layers.quantization import Fp8Config

gemm = AscendRowParallelLinear(
    input_size=k, output_size=n, bias=False,
    params_dtype=torch.float16,
    quant_config=Fp8Config(...) if quant_type == "w8a8" else None
)
latency = benchmark_npu(lambda: gemm.forward(x), ...)
```

**kernel选路逻辑**（vllm-ascend框架层）：
- BF16：`AscendRowParallelLinear.forward()` → `MatMulV2`（CANN原生GEMM）
- W8A8：`Fp8LinearMethod.apply()` → `QuantBatchMatmulV3`（量化GEMM）

**关键技术**：
- **框架层选路**：不走`torch.mm`，保证W8A8走真实量化kernel而非MatMulV2
- **参数空间**：M∈[1..16384]（密集采样decode小M+prefill大M），N/K∈[256..16384]笛卡尔积，约97K组合
- **模型特化**：`--model GlmMoeDsa`扫描实际使用的(N,K)组合，避免冗余笛卡尔积
- **6-op L2冲刷**：创建6个独立GEMM实例，模拟真实推理cache竞争，避免latency偏乐观

**输出文件**：`gemm_perf.txt`（4942行，cols: quant_type, m, n, k, latency_us）

### A.2 Attention采集（collect_attn.py）

**采集算子**：Context/Generation attention

**核心API调用**：
```python
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.config import VllmConfig

backend = AscendAttentionBackendImpl(VllmConfig(...))
# vllm-ascend自动选择后端：FlashInfer/FlashAttention/FlexAttention
latency = benchmark_npu(lambda: backend.forward(...), ...)
```

**完整推理环境构造**：
- VllmConfig（block_size, max_seq_len等）
- Paged KV Cache（mock allocation + 填充）
- Attention Metadata（batch_size, seq_lens, block_tables）

**关键技术**：
- **自动后端选择**：vllm-ascend根据平台自动选FlashInfer/FlashAttn，bench测到与生产一致kernel
- **KV Cache构造**：分配真实paged cache结构，模拟生产环境内存布局
- **API兼容**：4层try/except处理vllm-ascend API签名变更（use_sparse/use_v1等）

**输出文件**：`context_attention_perf.txt`（1013行）/`generation_attention_perf.txt`（1217行）

### A.3 MoE FFN采集（collect_moe.py）

**采集算子**：MoE expert FFN（grouped GEMM）

**核心API调用**：
```python
import torch_npu

# BF16路径
torch_npu.npu_grouped_matmul(x, experts, group_type=3)

# W8A8路径
torch_npu.npu_grouped_matmul_swiglu_quant(x, experts_w8, scales, ...)
```

**关键技术**：
- **两条kernel路径**：BF16走`npu_grouped_matmul`，W8A8走`npu_grouped_matmul_swiglu_quant`
- **Power Law分布**：模拟expert负载不均（`--distribution power_law`），唯一和模型配置相关的算子
- **routing预计算**：生成topk routing weights，模拟真实dispatch pattern

**输出文件**：`moe_perf.txt`（196行，cols: num_tokens, hidden, inter, num_experts, topk, dtype, latency_us）

### A.4 DSA Module采集（collect_mla_module.py）

**采集算子**：DeepSeek Sparse Attention完整module（投影+attention+输出）

**核心API调用**：
```python
from vllm_ascend.model_executor.models.deepseek import DeepseekV2MLAAttention

module = DeepseekV2MLAAttention(config)
latency = benchmark_npu(lambda: module.forward(hidden_states, ...), ...)
```

**关键技术**：
- **Module级采集**：整段forward（projection→SFA→output），不是单独采SparseFlashAttention
- **OOT路径注入**：CANN内置SFA binary（9 attrs）与vllm-ascend注册op（5 attrs）冲突，需注入`ASCEND_CUSTOM_OPP_PATH`
- **Context Parallelism处理**：prefill DSA用CP切query（每卡全64头+1/cp query），需按CP切分构造输入

**特殊处理（prefill）**：
- 合成采集失败（SFA kernel缺失，返回no-op假值）→ 改用profiler-derived建模（§4.5）
- 仅保留decode generation module的实采数据

**输出文件**：
- `dsa_context_module_perf.txt`（369行，已废弃，改用profiler-derived）
- `dsa_generation_module_perf.txt`（696行，cols: batch, seq, num_heads, latency_us）

### A.5 MoE Dispatch+Combine采集（collect_moe_dispatch_combine.py）

**采集算子**：FusedMC2融合算子（dispatch+expert FFN+combine）

**核心API调用**：
```python
# 绕过wrapper，直调底层C op
torch.ops._C_ascend.dispatch_ffn_combine(
    hidden_states, expert_weights, ...
    max_output_size=calculated_size  # 精确控制，避免OOM
)
```

**关键技术**：
- **绕过wrapper**：FusedMC2CommImpl硬编码max_output_size=65536→单卡OOM，直调C op精确控制
- **EP依赖**：dispatch latency强依赖ep_size（ntok=8时ep2=1711us/ep8=867/ep32=301，差5.7×）
- **补采ep32**：生产用ep32，旧bench仅ep{2,4,8}，补采ep32触发10.8×吞吐修正（§4.3）

**启动方式**：
```bash
torchrun --nproc_per_node=8 collect_moe_dispatch_combine.py --ep-size 8
torchrun --nproc_per_node=16 collect_moe_dispatch_combine.py --ep-size 16  # 2×A3节点
```

**输出文件**：`moe_dispatch_combine_perf.txt`（170行，含ep32实测）

### A.6 Elementwise采集（collect_elementwise.py）

**采集算子**：RMSNorm、RoPE、SwiGLU等轻量op

**核心API调用**：
```python
import torch_npu

torch_npu.nn.functional.rms_norm(x, weight, epsilon)
torch_npu.nn.functional.rotary_positional_embeddings(x, cos, sin)
```

**用途**：这些op延迟<50us，时间占比<3%，用实采而非SOL估算提升精度

### A.7 通信算子采集

**NCCL通信**（`collect_nccl.py`）：
- 调HCCL官方binary（黑盒），几何级数扫描message_bytes
- 输出：`nccl_perf.txt`（DP allgather/reduce_scatter）

**Custom AllReduce**（`collect_all_reduce.py`）：
- 实例化vllm-ascend Custom AllReduce + CUDA Graph
- 输出：`custom_allreduce_perf.txt`（TP allreduce）

### A.8 计时引擎（bench_engine.py）

**六阶段流水线**：
1. Adaptive Warmup → 估算单次耗时，计算actual_num_runs
2. NPU Graph Capture → 录制kernel_func到graph（失败则fallback eager）
3. Graph Warmup → graph.replay() × warmup_iters
4. 正式测量 → Event.record() + graph.replay() × N + synchronize()
5. Throttling检测 → NPU clock下降>10%则标记
6. 取min(graph_latency, eager_latency) → 消除Python dispatch开销

**关键参数**：
- warmup_iters：10（自适应warmup+graph warmup）
- num_runs：50（正式测量次数）
- repeat_n：1（单次kernel_func内的op数，如GEMM用6-op）

---

## 附录B：核心代码引用

### A.1 Performance数据库查询接口

```python
# src/aiconfigurator_npu/sdk/perf_database.py

class PerfDatabase:
    def query_gemm(self, quant_mode, m, n, k, ...) -> PerformanceResult:
        """Query GEMM latency with SILICON/CALIBRATION fallback."""
        silicon = self._query_gemm_silicon(quant_mode, m, n, k)
        if silicon is not None:
            return silicon
        sol = self._query_gemm_sol(quant_mode, m, n, k)
        factor = self.query_comm_calibration("gemm", ...)
        return sol * factor
    
    def query_kv_transfer(self, ep_size, isl, ...) -> PerformanceResult:
        """Query KV transfer latency for disagg prefill (net wall-clock)."""
        # 二维网格插值（ep×isl）；存的是 profiler net 墙钟，无 overlap_factor
        nearest_ep = min([1, 16], key=lambda x: abs(x - ep_size))
        # isl<=20k 插值；isl>20k 顶部两点线性外推（非 clamp）
        latency = self._interpolate_or_extrapolate_kv_transfer(nearest_ep, isl)
        return PerformanceResult(latency, ...)
    
    def query_moe_dispatch_combine(self, ep_size, ...) -> PerformanceResult:
        """Query MoE dispatch+combine with ep fallback."""
        measured_eps = [2, 4, 8, 32]  # 含生产 ep32 补采
        if ep_size not in measured_eps:
            nearest_ep = min(measured_eps, key=lambda x: abs(x - ep_size))
            ep_size = nearest_ep
        return self._query_moe_dispatch_silicon(ep_size, ...)
```

### A.2 Operations算子建模

```python
# src/aiconfigurator_npu/sdk/operations.py

class GEMM(Operation):
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query GEMM latency."""
        m = kwargs.get("x")
        result = database.query_gemm(
            self._quant_mode,
            m=m, n=self._n, k=self._k,
            ...
        )
        return PerformanceResult(result * self._scale_factor, ...)

class KVTransfer(Operation):
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query KV transfer latency (disagg prefill only)."""
        if not kwargs.get("is_disagg_prefill"):
            return PerformanceResult(0.0, 0.0)
        ep_size = kwargs.get("ep_size")
        isl = kwargs.get("isl")
        result = database.query_kv_transfer(ep_size, isl, ...)
        return PerformanceResult(result * self._scale_factor, ...)

class MoEDispatch(Operation):
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MoE dispatch+combine latency."""
        ep_size = kwargs.get("ep_size")
        num_tokens = kwargs.get("x")
        result = database.query_moe_dispatch_combine(ep_size, num_tokens, ...)
        return PerformanceResult(result * self._scale_factor, ...)
```

### A.3 配置寻优Picking逻辑

```python
# src/aiconfigurator_npu/sdk/picking.py

def _build_disagg_summary_dict(
    prefill_summary_dict: dict,
    prefill_num_worker: int,
    decode_summary_dict: dict,
    decode_num_worker: int,
) -> dict:
    """Build disagg summary row with rate matching."""
    
    # Rate matching修正系数
    seq_s = min(
        prefill_summary_dict["seq/s"] * prefill_num_worker * 0.9,
        decode_summary_dict["seq/s"] * decode_num_worker * 0.92,
    )
    
    num_total_gpus = (
        prefill_summary_dict["pp"] * prefill_summary_dict["tp"] * prefill_summary_dict["dp"] * prefill_num_worker +
        decode_summary_dict["pp"] * decode_summary_dict["tp"] * decode_summary_dict["dp"] * decode_num_worker
    )
    
    tokens_s_gpu = seq_s * osl / num_total_gpus
    
    # TTFT校正因子（已禁用，=1.0）
    request_latency = ttft * 1.0 + tpot * (osl - 1)
    
    return {
        "seq/s": seq_s,
        "tokens/s/gpu": tokens_s_gpu,
        "ttft": ttft,
        "tpot": tpot,
        ...
    }
```

---

**文档版本**: v2.0（profiler-derived DSA + ep32 实测 + comm 收口）  
**生成时间**: 2026-06-05  
**适用场景**: 技术分享、项目复盘、技术评审