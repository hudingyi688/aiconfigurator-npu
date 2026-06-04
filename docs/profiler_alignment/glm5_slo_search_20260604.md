# GLM-5 SLO 配置寻优结果（profiler-derived DSA 修正版）

状态：2026-06-04（取代 2026-05-30 版）
部署：2P2D / 4 节点 / 64 卡 Ascend 910B，vllm-ascend 0.18.0，GLM-5-w8a8
工具：`aic-npu`（HYBRID 模式，chunked-prefill）
SLO：osl=2500，P90 TPOT < 70ms，P50 TTFT 分档

## ⚠️ 本版相对 05-30 版的重大修正

05-30 版结论「DSA 是 prefill 绝对主导（占 64%→92%）」**已作废**。该结论基于错误的 DSA 口径：
用了 `num_heads=4` 的合成 silicon，且**没有按 Context Parallelism 切分 query**
（把每卡 query 当成完整 chunk 4096，而生产 CP16 下每卡只处理 256），把单请求 DSA
放大成 3315ms（isl=10k）。

修正后（profiler-derived DSA，单 rank、CP、nh=64，已端到端锚定到生产 profiler）：

- **prefill 主导项重新是 KV transfer，不是 DSA**。isl=10k 时 DSA 仅占 prefill 8%，
  KV transfer 占 82%。这与最早「KV transfer 占 prefill ~87%」的结论一致——05-30 版
  说推翻了它，其实那次「推翻」本身是基于错误的 DSA 数据。
- DSA 单卡 162ms（isl=10k）与生产 profiler 实测单卡 DSA 注意力核心 152.5ms 一致
  （+6%，差异为投影 GEMM），**端到端锚定成功**；profiler 自身也是
  KV transfer(1679ms) >> DSA(152ms)，双向印证。

## 结论速览

- prefill 主导项 = **KV transfer**（isl≤20k 实测；DSA 退居次要，占 8%→23%）。
- DSA 现为 **profiler-derived**：单卡 attention 核心按累积 KV 查实测表（SFA 在
  KV≥8192 sparse topk 封顶、indexer ∝KV 线性），投影 GEMM 走解析 SOL；每卡 query
  按 CP 切分（query = chunk / cp），末 chunk 按实际 query 比例缩放。
- 降 prefill TTFT 的重点从 05-30 版的「减 DSA 重算」**改回「减 KV transfer」**。

## 各档 prefill 分解（disagg prefill，单请求，profiler-derived DSA）

> HYBRID 模式。TTFT 为单请求 prefill（未叠加并发的排队修正）。DSA、KV transfer
> 均为单请求 / 80 层聚合口径，可比。

| 档 | isl | prefill 配置 | prefill TTFT | DSA | KV transfer | MoE+other | DSA 占比 | 可信度 |
|---|---|---|---|---|---|---|---|---|
| 档1 | ~10k | tp16/ep32/dp2 | **2049ms** | 162ms | 1679ms | 208ms | 8% | 🟢 |
| 档2 | ~20k | tp16/ep32/dp2 | **3305ms** | 428ms | 2571ms | 306ms | 13% | 🟢 |
| 档3 | ~40k | tp32/ep32/dp1 | **3549ms** | 485ms | 2571ms* | 493ms | 14% | 🟡 |
| 档4 | ~80k | tp32/ep32/dp1 | **4473ms** | 1029ms | 2571ms* | 873ms | 23% | 🟡 |

> **\* KV transfer 在档3/4 被 clamp**：KV-transfer 实测表仅标定到 isl=20k，isl>20k
> 持平在 20k 值（2571ms），是下界近似。档3/4 的 prefill 总值因此**偏低**（真实
> KV transfer 随 isl 增长），标 🟡。DSA 部分仍可信（表覆盖到 KV=20480，且 SFA
> 已饱和、indexer 线性外推稳定）。

## DSA 的端到端锚定（本版核心证据）

| isl | 模型 DSA 单卡 | profiler 实测单卡 DSA 核心 | 偏差 |
|---|---|---|---|
| 10k | 162ms（核心 144 + 投影 18） | 152.5ms | +6% |
| 20480 | 428ms（核心 392 + 投影 36） | 392.3ms | +0.0%（核心） |

profiler 数据来自生产单请求 prefill kernel timeline（dp1/ep16，CP16，nh=64），
10k 与 20k 在相同累积 KV 点交叉验证 <4%。10k 末 chunk query 实测仅 114（不满
256），模型按 query 比例缩放后核心 144ms（profiler 152.5，−5.7%）。

## TPOT（decode 侧）—— 沿用既有结论（本次未改 decode）

decode TPOT 模型经双 batch（batch=1 / batch=7）stream 对账，确认**模型基本准确**
（batch=1 预测 52.7ms vs 实测墙钟 53.0ms，−0.6%）：attention 与 MoE 在硬件上完全
串行（两 batch 点 attn_union + moe_union = compute 合并 union），模型串行相加结构
正确，无需并行折扣。05-30 版「TPOT 高估 1.6×、根因 module 间并行未建模」的结论
**已作废**（那是错误对账的产物）。

> 唯一遗留：KV pool（AICPU batch_get）在高 batch 若超过 compute union 会露头，
> 当前藏在 compute 后，模型不建它暂时安全。

## 各档 GAP 分析（prefill，isl≤20k 可信区）

| 档 | prefill | 假想 SLO | 主因 |
|---|---|---|---|
| 档1 | 2.0s | 2s | KV transfer 1.68s 主导（占 82%）|
| 档2 | 3.3s | 5s | KV transfer 2.57s 主导（占 78%）|

> **核心 GAP**：prefill 由 **KV transfer 主导**（isl≤20k 占 78%~82%）。降 prefill
> TTFT 的重点是减 mooncake KV transfer（连接器 / 带宽 / 重叠），而非减 DSA 重算。
> DSA 虽随 isl 增长（每 step 重算累积 KV 的 sparse attention），但 sparse topk 封顶
> 使其增长受限，绝对值远小于 KV transfer。

## SLO 寻优最优配置（64 卡，osl=2500，TPOT<70ms）

用 `aic-npu default`（HYBRID）跑 4 档 SLO 配置搜索，各档吞吐最优配置：

| 档 | isl | TTFT 约束 | 最优模式 | 并行配置 | TTFT | TPOT | 单卡 tok/s | 单用户 tok/s |
|---|---|---|---|---|---|---|---|---|
| 档1 | 10k | <2000ms | agg | tp4/dp8/ep32 | 1937ms | 43.1ms | **5.75** | 23.2 |
| 档2 | 20k | <5000ms | agg | tp4/dp16/ep64 | 4321ms | 43.2ms | **5.67** | 23.2 |
| 档3 | 40k | <8000ms | agg | tp8/dp8/ep64 | 5101ms | 42.5ms | **2.87** | 23.5 |
| 档4 | 80k | <10000ms | — | **无解** | — | — | — | — |

> 档4：isl=80k 在 64 卡上**放不下**（模型权重 + KV cache 超 HBM，与 SLO 无关，放宽
> SLO 仍无解）；需更多卡。

### 关键结论

1. **最优全是 agg（聚合）模式，disagg（PD 分离）在这些 SLO 下不划算**：档1/档3
   disagg 无可行解，档2 disagg 仅 0.44× agg 吞吐。原因正是本版的核心——prefill 由
   KV transfer 主导（1.7~2.6s），PD 分离把 KV 流式传输的开销暴露在关键路径上，反而
   被 agg（无跨节点 KV transfer）超过。
2. **TPOT 全档 ~43ms，远达标（<70）**。这推翻 05-30 版「TPOT ~87ms 超标」——印证
   decode 模型修正后 TPOT 准确（batch=1 实测对账 −0.6%）。TTFT 全档达标。
3. 单卡吞吐随 isl 升高而降（5.75→2.87 tok/s/gpu），因长序列 prefill/KV 占比上升。

> 复现：`aic-npu default --model zai-org/GLM-5 --system ascend_910b --backend
> vllm-ascend --backend-version 0.18.0 --database-mode HYBRID --total-gpus 64
> --isl <ISL> --osl 2500 --ttft <TTFT> --tpot 70`。
> （报告末尾的 "generator artifact" 警告是 stub 禁用，无害；搜索本身成功。）

## 已知不可信区 / 待办

1. **isl>20k 的 KV transfer**：表仅到 20k，档3/4 持平为下界近似 → prefill 总偏低。
   需补采 isl=40k/80k 的 KV-transfer profiler 才能让档3/4 数值可信。
2. **architecture dims 错配**：GLM-5 config 声明 `DeepseekV32ForCausalLM`，但实际
   dims（hidden6144/q_lora2048/v256/idx32）匹配 `GlmMoeDsaForCausalLM` 条目。投影
   SOL 当前用 DeepseekV32 dims，影响投影绝对值（~百 us/层级，占 DSA 小头）。DSA
   attention 核心是实测值，不受影响。
3. **配置形态结论**：tp/dp/ep 形态选择、同部署族内相对排名可信；isl≤20k 的
   prefill 绝对 TTFT 已端到端锚定（DSA 侧）；isl>20k 待 KV transfer 补采。
