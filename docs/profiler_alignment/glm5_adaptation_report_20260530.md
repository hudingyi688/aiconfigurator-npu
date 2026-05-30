# AIConfigurator-NPU 适配 GLM-5 报告

状态：2026-05-30
分支：`local/dsa-debug-snapshot`
适用对象：GLM-5-w8a8 / Ascend 910B / vllm-ascend 0.18.0

> 本报告遵循 YAGNI 原则：只陈述已实现、已验证、有数据支撑的内容，不为完整性而堆砌设想。
> 所有 file:line 引用均对应仓库当前代码，数字均来自实测采集或生产 profiler。

---

## 目录

1. AIConfigurator 的设计 —— 数据采集机制与 vllm-ascend API 调用
2. GLM-5 算子分析 —— 算子→采集方式映射，及待改进项
3. 与 profiler 的算子+shape 对齐 —— 含调用次数
4. 配置寻优方式与结果 —— 与生产配置的对齐

---

## 1. AIConfigurator 的设计

### 1.1 核心范式：算子级硅数据 + 解析模型

AIConfigurator 的设计前提是 **「系统延迟 = 各算子延迟之和」**。它不跑完整推理服务，而是：

1. **采集（collect）**：在真机 NPU 上，直接构造单个算子的输入张量、单独计时该 kernel，得到 `(shape, dtype, 并行度) → latency` 的硅数据表。
2. **建模（model）**：把一个 GLM-5 layer 拆成有序的 Operation 序列（GEMM / Attention / MoE / MoEDispatch / 通信 …），每个 Operation 在估算时查硅表 + 解析插值。
3. **寻优（search）**：在 tp/dp/ep/batch 等并行维度上扫描，对每个配置累加算子延迟得到 TTFT/TPOT，在 SLA 约束下挑吞吐最优配置。

与 msmodeling 这类「整服务 profiler」方法的本质区别：采集成本低、跨配置可解析插值（不必每个配置重跑），但**对调度器行为不可见**（见第 4 节 KV transfer 缺口）。

### 1.2 采集机制

采集脚本位于 `collector/npu/`，统一通过 `collector/bench_engine.py:benchmark_npu()` 计时：

- **直接 import** `torch_npu` + `vllm_ascend` 的 kernel，构造输入张量，单独调用目标算子（`bench_engine.py:58-118`）。
- **计时**：warmup 20 次后，用 `torch.npu.Event` 对包住 N 次 replay（默认 `num_runs=100`），`latency = total / num_runs / repeat_n`。
- **NPU Graph**：优先在非默认 stream 上捕获 graph 并 replay；同时跑 eager，**取两者较小值**；graph 捕获失败自动降级 eager（`bench_engine.py:87-118`）。

### 1.3 各采集脚本调用的 vllm-ascend / torch_npu API

| 脚本 | 采集算子 | 调用的 API | 扫描维度 | 输出 CSV |
|---|---|---|---|---|
| `collect_gemm.py` | BF16 / W8A8 GEMM | `vllm_ascend.ops.linear.AscendRowParallelLinear` → `MatMulV2` / `QuantBatchMatmulV3` | M×{N,K} 笛卡尔积 + 模型特定对 | `MatMulV2.csv` / `QuantBatchMatmulV3.csv` |
| `collect_attn.py` | Context / Decode attention | `vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl.forward` | batch×seq×heads×kv_heads | `FusedInferAttentionScore[_Decode].csv` |
| `collect_moe.py` | MoE FFN (含 group GEMM) | `npu_grouped_matmul` / `npu_grouped_matmul_swiglu_quant` (经 `moe_factory`) | num_tokens∈[1..4096] × 9 个模型配置 | `GroupedMatmul_MoE_{BF16,W8A8}.csv` |
| `collect_mla.py` / `collect_mla_module.py` | MLA / DSA module 前向 | `npu_fused_infer_attention_score[_v2]`；module 级整段 forward | batch×seq×heads | `dsa_{context,generation}_module_perf.txt` |
| `collect_elementwise.py` | rmsnorm/rope/swiglu/softmax | `torch_npu.npu_*`（直调，不走 wrapper） | batch×hidden×inter×heads | `elementwise_perf.csv` |
| `collect_moe_dispatch_combine.py` | FusedMC2 融合 dispatch+FFN+combine | `torch.ops._C_ascend.dispatch_ffn_combine`（**绕过 wrapper**） | ep{2,4,8}×dtype×num_tokens | `moe_dispatch_combine_ep{ep}.csv` |

### 1.4 一个关键的采集设计决策：绕过 wrapper

`collect_moe_dispatch_combine.py` 刻意**绕过** vllm-ascend 的 `FusedMC2CommImpl`，直接调底层 C op `dispatch_ffn_combine`（`moe_dispatch_factory.py:10-19`）。原因：wrapper 硬编码 `max_output_size=65536`，在合成的单卡/双卡/16 卡 bench 上会分配 9-12 GiB 最坏情况 scratch 而 OOM；裸 C op 可把 `max_output_size` 精确设为 `ceil(M×K/NL)+margin`。其余脚本（gemm/attn/moe）均走 vllm-ascend 框架级 API，保证与生产 kernel 选路一致。

### 1.5 数据落地

采集产物按 `KERNEL_TYPE_MAP`（如 `collect_gemm.py:124`）命名，最终进入
`systems/data/ascend_910b/vllm-ascend/0.18.0/`，由 `PerfDatabase` 按 `PerfDataFilename` 枚举加载。当前已就位的硅表：

```
gemm_perf.txt / MatMulV2.csv / QuantBatchMatmulV3.csv     (GEMM)
context_attention_perf.txt / generation_attention_perf.txt (Attention)
moe_perf.txt / GroupedMatmul_MoE_{BF16,W8A8}.csv           (MoE FFN)
dsa_{context,generation}_module_perf.txt                   (DSA module)
moe_dispatch_combine_perf.txt                              (FusedMC2 融合算子)
kv_transfer_perf.txt                                       (PD 分离 KV transfer)
custom_allreduce_perf.txt / nccl_perf.txt                  (通信)
comm_calibration.json                                      (通信校准系数)
```

<!-- SECTION2 -->

## 2. GLM-5 算子分析与采集方式

### 2.1 GLM-5 模型结构（来自 `model_configs/zai-org--GLM-5_config.json`）

| 维度 | 值 | 维度 | 值 |
|---|---|---|---|
| num_hidden_layers | 78 | hidden_size | 6144 |
| num_attention_heads | 64 | kv_lora_rank | 512 |
| qk_nope/rope_head_dim | 192 / 64 | v_head_dim | 256 |
| num_experts (routed) | 256 | num_experts_per_tok (topk) | 8 |
| moe_intermediate_size | 2048 | n_shared_experts | 1 |
| first_k_dense_replace | 3 | index_topk (DSA) | 2048 |

GLM-5 是 671B 级 MoE，注意力为 **DSA（DeepSeek Sparse Attention，含 LightningIndexer + SparseFlashAttention）**，架构标识 `DeepseekV32ForCausalLM` → 走 `DeepSeekV32Model`（`models.py:364`，vllm-ascend 非 wideep 分支）。

### 2.2 算子 → 采集方式映射

一个 GLM-5 layer 在 `DeepSeekV32Model` 里被拆为下列 Operation，各自的硅数据来源：

| 算子类别 | Operation | 采集脚本 / 数据源 | 数据状态 |
|---|---|---|---|
| QKV / O / dense FFN GEMM | `GEMM` | `collect_gemm.py` → `gemm_perf.txt` | ✅ silicon |
| DSA 注意力（含 indexer/SFA/投影） | `ContextDSAModule` / `GenerationDSAModule` | `collect_mla_module.py` → `dsa_*_module_perf.txt` | ⚠️ silicon 仅 num_heads=64（TP=1） |
| MoE 路由 FFN（专家 group GEMM） | `MoE` | `collect_moe.py` → `moe_perf.txt` | ✅ silicon |
| MoE dispatch / combine（EP 通信） | `MoEDispatch` | `collect_moe_dispatch_combine.py` → `moe_dispatch_combine_perf.txt` | ✅ silicon（FusedMC2，ep{2,4,8}） |
| rmsnorm / rope / swiglu | `ElementWise` | `collect_elementwise.py` | ✅ silicon |
| TP allreduce | `CustomAllReduce` | `custom_allreduce_perf.txt` | ✅ silicon |
| DP allgather / reducescatter / a2a | `query_nccl` | `nccl_perf.txt` + `comm_calibration.json` | ⚠️ 部分 calibration |
| PD 分离 KV transfer（prefill→decode） | `KVTransfer` | profiler 反推 → `kv_transfer_perf.txt` | ⚠️ 仅 prefill，ep{1,16}×isl{2500,10k,20k} |

### 2.3 当前存在的不足与改进项

按对配置寻优的影响排序：

1. **DSA 模块只有 TP=1（num_heads=64）的硅数据**（最高优先）
   生产 prefill tp=16 → num_heads_per_rank=4、decode tp=4 → 16，硅表只有 64。SILICON 模式下 `query_*_dsa_module` 直接 Qhull 报错崩溃，只能用 HYBRID（SOL+经验回退）。
   改进：`mla_module_factory.py` 已支持 `--num-heads-override`，需在 NPU 上重采 TP∈{2,4,8,16} → num_heads∈{32,16,8,4}，追加进 `dsa_*_module_perf.txt`，即可解锁纯 SILICON 寻优。

2. **MoE dispatch/combine 融合表只覆盖 ep{2,4,8}**
   生产 prefill 用 ep16。已做缓解：`query_moe_dispatch_combine` 对未测 ep 夹取到最近已测 ep（hold flat，`perf_database.py`），使 ep16 不再崩溃。
   改进：在 NPU 上补采 ep16（及 ep32）的 `dispatch_ffn_combine` 数据。

3. **MoE dispatch BF16 路径仍走 calibration 而非 silicon**
   decode 阶段 `MoeDistributeDispatchV2/CombineV2`（BF16）目前用 `comm_calibration.json` 系数近似。
   改进：新增 `collect_moe_dispatch_bf16.py`，把 BF16 路径也提升为 silicon。

4. **若干小算子未单独建模**（低优先，合计 <5% 时间占比）
   `mla_preprocess`、`MoeGatingTopK`、`ReshapeAndCache`、RoPE 等可分离小 kernel 目前并入 SOL 估算。按 YAGNI，对寻优影响可忽略，暂不单独采集。

<!-- SECTION3 -->

## 3. 与 Profiler 的算子 + Shape 对齐（详细）

数据源：11 个 GLM-5 生产 profiler run（`glm5-profiler.tar.gz`，prefill/decode × dp/ep 配置），聚合为 `docs/profiler_alignment/groundtruth/`：
- `detail.csv`：3203 个 (run, phase, isl, tp/dp/ep, op_type, shape) 条目，882413 次 op 调用
- `by_op_family.csv`：1722 个 (family, sub_op, shape) 条目，已按 family 归类
- 对齐工具：`tools/check_alignment.py`（profiler 中位 vs bench 实测，纯 CSV 解析）

### 3.0 度量口径说明（重要）

profiler 的 `avg_us` 在小 isl 下被 **10-20 秒级 rank 同步等待气泡**严重污染（KV transfer 族 max 达 2×10⁷ us）。因此本节区分两个口径：

- **avg×count（含气泡）**：等于墙钟时间份额，是 `coverage_analysis_20260528.md` 的口径。KV transfer 占 prefill **87.4%**。
- **median×count（去气泡）**：真实 device 计算份额。KV transfer 占 prefill **34.3%**。

两个数字都正确，口径不同。下文覆盖率表**两者并列给出**。算子级偏差对齐统一用 **median**（稳定、抗离群）。

### 3.1 Profiler 按总时长 Top 算子（含调用次数）

| op_type | unique shapes | 调用次数 | total_ms | avg_us/call | aic-npu 来源 |
|---|---:|---:|---:|---:|---|
| QuantBatchMatmulV3 | 69 | 87681 | 61542 | 701.9 | silicon (gemm) / 部分 MoE expert-batched |
| DispatchFFNCombine | 8 | 19950 | 61300 | 3072.7 | silicon (moe_dispatch_combine) |
| broadcastAicpuKernel | 1 | 2964 | 59940 | 20222.6 | KVTransfer（新增） |
| reduce_scatterAicpuKernel | 1 | 2464 | 52283 | 21218.9 | KVTransfer（新增） |
| hcom_broadcast_ | 1 | 2964 | 32346 | 10913.1 | KVTransfer（新增） |
| hcom_reduceScatter_ | 1 | 2616 | 22786 | 8710.2 | KVTransfer（新增） |
| SparseFlashAttention | 17 | 20399 | 3122 | 153.0 | silicon (DSA module) |
| LightningIndexer | 17 | 20400 | 1281 | 62.8 | silicon (DSA module 内) |
| MatMulV2 | 72 | 62430 | 1236 | 19.8 | silicon (gemm BF16) |
| hcom_allReduce_ | 1 | 74786 | 1028 | 13.8 | silicon (custom_allreduce) |

### 3.2 GEMM 算子级偏差对齐（`tools/check_alignment.py` 全量结果）

对所有有 bench 对应的 (op_type, shape) 逐条比对 profiler 中位 vs bench：

**整体分布（real-signal，kernel ≥ 30us，共 71 条）**

| 指标 | 值 |
|---|---|
| 平均偏差 | -11.2% |
| 中位偏差 | -14.0% |
| 平均 \|偏差\| | 27.7% |
| 最大 \|偏差\| | 224.4% |

偏差分桶：`[0,10)% → 17 条`，`[10,20)% → 24 条`，`[20,50)% → 20 条`，`[50,100)% → 9 条`，`[100,∞)% → 1 条`。
另有 14 条 small-op（<30us，~1% 时间占比）+ 56 条 bench MISS（该 shape 未采）。

**W8A8 (QuantBatchMatmulV3) 对齐良好样本（调用次数高、偏差小）**

| M | N | K | dtype | profiler调用 | profiler中位(us) | bench(us) | 偏差 |
|--:|--:|--:|---|--:|--:|--:|--:|
| 9 | 1024 | 6144 | w8a8 | 3619 | 36.86 | 41.71 | +13.2% OK |
| 6 | 1024 | 6144 | w8a8 | 3542 | 35.49 | 42.50 | +19.8% OK |
| 256 | 6144 | 2048 | w8a8 | 2772 | 52.11 | 46.41 | -10.9% OK |
| 256 | 4096 | 2048 | w8a8 | 2080 | 47.89 | 46.75 | -2.4% OK |
| 256 | 16384 | 2048 | w8a8 | 2080 | 87.18 | 70.32 | -19.3% OK |
| 114 | 6144 | 2048 | w8a8 | 154 | 40.76 | 39.66 | -2.7% OK |

**BF16 (MatMulV2) 对齐良好样本**

| M | N | K | profiler调用 | profiler中位(us) | bench(us) | 偏差 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 38720 | 6144 | 218 | 393.09 | 363.96 | -7.4% OK |
| 3 | 6144 | 12288 | 218 | 143.41 | 123.30 | -14.0% OK |
| 3 | 6144 | 6144 | 218 | 64.35 | 62.77 | -2.5% OK |
| 3 | 38720 | 6144 | 203 | 397.93 | 363.88 | -8.6% OK |

**偏差较大样本（已分析根因，不影响寻优）**

| op | M | N | K | profiler中位 | bench | 偏差 | 根因 |
|---|--:|--:|--:|--:|--:|--:|---|
| QuantBatchMatmulV3 | 18 | 1024 | 6144 | 131.72 | 39.61 | -69.9% FAIL | expert-batched 误归类（见 3.3） |
| MatMulV2 | 21 | 6144 | 6144 | 84.29 | 66.07 | -21.6% WARN | TP 切分边界 |
| MatMulV2 | 12 | 6144 | 12288 | 154.14 | 120.93 | -21.5% WARN | dense ffn2 |
| (256,256,6144) | — | — | — | 19.6 | 46.7 | +138% | router 小 op，bench loop 高估 ~30us |

结论：real-signal GEMM 中位偏差 -14%、61/71 条在 ±50% 内。偏差主要来自 TP 切分边界和小 op 的 bench dispatch overhead，绝对值占比小，对 Pareto 排名无实质影响。

### 3.3 一个关键的 shape 归类陷阱（已澄清）

profiler 里有 539 个 `aclnnQuantMatmulWeightNz` shape 形如 `(256,4096,6144)`，profiler 中位 **2055us** 而 bench 单 op 仅 62us（-97%）。根因：被 NPU runtime 归到 `QuantBatchMatmulV3` op type 下的这批，**实际是 MoE 内部 expert-batched W8A8 group GEMM**（一次调用做 N 个 expert），延迟是单 op 的几十倍。
这条数据**不参与寻优**——MoE 路径查 `moe_perf.txt`，不查 `gemm_perf.txt`。check_alignment 里那条 -69.9% FAIL 同源，已知非缺陷。

### 3.4 DSA 注意力与通信对齐

| profiler kernel | 调用次数 | 中位(us) | aic-npu 对应 | 状态 |
|---|--:|--:|---|---|
| SparseFlashAttention | 20399 | 153.0 | `ContextDSAModule` 内 | module 级采集，语义对齐 |
| LightningIndexer | 20400 | 62.8 | DSA module 内 indexer | 调用次数与 SFA 一致 |
| mla_preprocess_0_mix_aic | 17784 | 69.2 | 并入 module forward | 不单独采 |
| hcom_allReduce_ | 74786 | 13.8 | `custom_allreduce_perf.txt` | silicon，单次极小 |
| hcom_allGather_ | 41407 | 35.7 | `query_nccl(all_gather)` | silicon |
| hcom_alltoall_ | 4324 | 382.3 | calibration (a2a SOL) | 校准近似 |

DSA module 在 bench 和 profiler 两侧都是 module 级（投影+attention+输出），语义对齐，无需逐 sub-kernel 验证；唯一缺口是 TP>1（num_heads≠64）的 silicon 数据（见 2.3）。

### 3.5 总覆盖率（两口径）

按 profiler device-time 份额分类（每个 op_type 归到 aic-npu 来源），两口径并列：

**Prefill 阶段**

| 来源 | median×count（去气泡） | avg×count（含气泡=墙钟份额） |
|---|---:|---:|
| silicon | 33.5% | — |
| calibration | 26.1% | — |
| SOL/elementwise | 1.8% | — |
| **KV transfer**（本轮新建模） | **34.3%** | **87.2%** |
| unmodeled_misc | 4.2% | — |
| **可建模合计（silicon+calib+SOL+KVTransfer）** | **95.7%** | — |

> prefill 两口径差异完全来自 KV transfer 的同步气泡：avg 口径下 KV transfer 吞掉 87% 墙钟（rank 间等待计入传输 kernel），median 口径下其真实 device 计算占 34%。覆盖度文档（`coverage_analysis_20260528.md`）用的是 avg/墙钟口径，故记为 87%。两者一致、不矛盾。**本轮 KVTransfer 建模后，prefill 在两口径下都已从「未覆盖」转为「已建模」。**

**Decode 阶段**

| 来源 | median×count |
|---|---:|
| silicon | 48.0% |
| calibration | 25.9% |
| SOL/elementwise | 12.7% |
| unmodeled_misc | 13.4% |
| **可建模合计** | **86.6%** |

> decode 的 unmodeled_misc（13.4%）主要是 PadV3/MemSet/ScatterNdUpdate 等内存搬运小 kernel 和 sampling，分散且单个占比低；TP>1 DSA 由 HYBRID 的 SOL+经验补。decode 绝对延迟实测在 ±25% 内（`coverage_analysis_20260528.md`）。

### 3.6 对齐状态汇总

| 算子族 | 对齐状态 | 寻优精度影响 |
|---|---|---|
| BF16 GEMM 中大 op | ✅ 中位偏差 -14% | 低 |
| BF16/W8A8 GEMM 小 op (router) | ⚠️ +138% | 极低（<50us，占比<1%） |
| W8A8 GEMM 单 op | ✅ 多数 ±20% | 可接受 |
| W8A8 expert-batched | 走 moe_perf.txt | 不经 GEMM 表 |
| MoE FusedMC2 dispatch | ✅ silicon (ep{2,4,8}) | 生产 ep16 靠夹取 |
| DSA module | ✅ 语义对齐 | TP>1 数据缺，靠 HYBRID |
| 通信 allreduce/allgather | ✅ silicon | 低 |
| a2a | ⚠️ calibration | 中 |
| KV transfer | ✅ 本轮建模 | prefill 关键 |

<!-- SECTION4 -->

## 4. 配置寻优方式与结果

### 4.1 寻优流程


入口 `task.py:build_disagg_parallel_lists` 定义搜索空间 → `InferenceSession` 静态估算每个候选 → `picking.py` 聚合并挑选 Pareto 最优。

**搜索维度**（vllm-ascend MoE 分支，`task.py:228-252`）：
- prefill tp ∈ {1,2,4,8,16}、decode tp ∈ {1,2,4,8}
- moe_ep ∈ {1,2,4,8,16,32,64}、dp 随 `tp×dp == moe_tp×moe_ep` 约束派生
- moe_tp 固定 = 1（vllm-ascend 不支持 MoE-TP 与 MoE-EP 同时 >1）
- batch_size、num_workers 扫描

**目标与约束**：在 TTFT ≤ 3000ms（prefill 4096）、TPOT ≤ 50ms 约束下，最大化 `tokens/s/gpu`（`picking.py` Pareto）。

### 4.2 Disagg 的 rate matching（`picking.py:73-94`）

prefill 与 decode 独立搜索后，按吞吐速率匹配：

```python
seq_s = min(
    prefill_seq_s * prefill_num_worker * 0.9,   # 0.9 = pipeline bubble / 排队
    decode_seq_s  * decode_num_worker  * 0.92,  # 0.92 = decode batch slot 未饱和
)
tokens_s_gpu = seq_s * osl / num_total_gpus
request_latency = ttft + tpot * (osl - 1)
```

三个修正系数的语义（彼此正交）：
- `_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9`：prefill 管道气泡。
- `_RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92`：decode batch slot 未饱和。
- `_AUTOSCALE_TTFT_CORRECTION_FACTOR = 1.8`（`picking.py:38`）：**并发排队修正**，非 KV transfer。公式 `lc/20+0.95`（N=10 批、本地并发 lc=15-20）。作用在单请求 ttft 上，建模多请求竞争同一 prefill worker 的平均排队。

### 4.3 与生产配置的对齐（commit `6849445`）

**生产部署形态**（GLM-5-w8a8，32 卡 PD 分离，启动脚本确认）：
- prefill：tp=16、dp=2、ep=32（一个 A3 host）
- decode：tp=4、dp=8、ep=32
- chunked prefill `max-num-batched-tokens=4096`、prefix-caching 开、Mooncake P2P KV、MTP=2

**对齐做法与结果**：无约束搜索在 64 卡上返回 `prefill tp=2 dp=16`——数学上 Pareto 最优，但 aic-npu 内存模型对低 TP 过于乐观（把 expert 权重均摊到 ep rank，忽略每 rank 的 KV pool / 激活开销），运行时无法装入 HBM。因此 `task.py:250-252` 把 DEEPSEEKV32 族 pin 到生产验证形态 prefill tp=16 / decode tp=4。pin 后：

```
disagg top-1:  prefill tp=16 dp=2 ep=32   ← 与生产一致
               decode  tp=4  dp=8 ep=32   ← 与生产一致
```

**配置形态完全复现生产**（commit `6849445` 验证）。

### 4.4 当前可信 / 不可信边界（`coverage_analysis_20260528.md`）

算子覆盖度（按 profiler **墙钟时间份额** = avg×count 口径，与 `coverage_analysis_20260528.md` 一致）：

| 阶段 | silicon | calibration | 未覆盖 |
|---|---|---|---|
| prefill | 6.6% | 4.0% | **87.2%**（KV transfer，本轮已新增建模） |
| decode | 47.4% | 46.9% | 5.7%（TP>1 DSA + misc） |

> 注：此表是**墙钟口径**（含同步气泡），与 §3.5 的 device 计算口径（median×count）数字不同但不矛盾——见 §3.0 口径说明。本轮 KVTransfer 建模后，prefill 那 87.2% 已从「未覆盖」转为「已建模」。


**可信**：
- 配置形态（tp/dp/ep、agg vs disagg）——pin 后复现生产。
- 同部署族内的相对排名（A 比 B 快 30% 这类结论可信）。
- decode 绝对延迟（±25%，覆盖 94%）。

**此前不可信、本轮改善**：
- prefill 绝对吞吐曾系统性低估 ~50%，根因是 KV transfer（占 prefill 87%）未建模。**本轮已新增 `KVTransfer` 算子**（profiler wall-clock trace 反推、overlap_factor=1.0 标定），让 prefill ttft 自带真实 KV transfer 成本。详见 `kv_transfer_perf.txt` 与 `query_kv_transfer`。

**仍不可信**：
- 长上下文（isl>8k）：KV transfer 网格只标定到 20k，且 chunked-prefill 调度未建模。
- agg vs disagg 直接对比：agg 无 KV transfer 可减，对比口径不对等。

### 4.5 本轮适配新增（2026-05-30，commits `a741b6c` + `fbc6242`）

1. **FusedMC2 融合 MoE 算子硅表**：`moe_dispatch_combine_perf.txt`，替代原 a2a 解析校准；门控放宽至覆盖 dp>1 的 DP-attn+EP-MoE 主流形态。
2. **PD 分离 KV transfer 建模**：`kv_transfer_perf.txt` + `KVTransfer` 算子，填补 prefill 最大覆盖缺口（87%）。门控 `is_disagg_prefill`，仅 disagg prefill 计费，agg/decode 返回 0。
3. **ep 越界夹取修复**：`query_moe_dispatch_combine` 对未测 ep（如生产 ep16）夹取到最近已测 ep，避免 disagg 搜索崩溃。

测试：54 项全通过。两个 commit 已推送至 `origin/local/dsa-debug-snapshot`。

---

## 附：后续优先级（YAGNI 取舍）

只列已识别、有明确收益的项，不含设想：

1. **NPU 上重采 DSA module TP∈{2,4,8,16}**（解锁纯 SILICON 寻优，消除 HYBRID 依赖）。
2. **补采 MoE dispatch ep16/ep32**（生产 prefill 用 ep16，当前靠夹取近似）。
3. **KV transfer overlap_factor 用真实 bench TTFT 复核**（当前 trace 标定=1.0，bench TTFT 未保存；若后续有 P50 可二次校准）。

不做（无必要）：小算子（mla_preprocess/RoPE/GatingTopK）单独采集——合计 <5% 占比，并入 SOL 已足够。



