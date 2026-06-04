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

## SLO 寻优最优配置（64 卡，osl=2500，TPOT<70ms；**ep32 实测后**）

用 `aic-npu default`（HYBRID）跑 4 档 SLO 配置搜索。**本表为补采生产 ep32
moe_dispatch silicon 后的结果**（见下方"ep32 修正"），各档吞吐最优配置：

| 档 | isl | TTFT 约束 | 最优模式 | 并行配置 | bs/并发 | TTFT | TPOT | 单卡 tok/s | 单用户 tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 档1 | 10k | <2000ms | agg | tp8/dp8/ep64 | 16/128 | 1980ms | 31.2ms | **62.0** | 32.0 |
| 档2 | 20k | <5000ms | agg | tp8/dp8/ep64 | 11/88 | 3426ms | 33.6ms | **38.9** | 29.7 |
| 档3 | 40k | <8000ms | agg | tp8/dp8/ep64 | 2/16 | 5505ms | 26.1ms | **9.2** | 38.3 |
| 档4 | 80k | <10000ms | — | **无解** | — | — | — | — | — |

> 档4：isl=80k 在 64 卡上**放不下**（模型权重 + KV cache 超 HBM，与 SLO 无关，放宽
> SLO 仍无解）；需更多卡。
> 注：ep64 配置在 dispatch 表里 snap 到最近的 ep32（grid {2,4,8,32}）。

### ep32 修正（重大，10.8× 单卡吞吐）

补采前 dispatch silicon 只有 ep2/4/8，生产 ep32/ep64 **clamp 到 ep8**，把 dispatch
高估 ~16%（ep8=405us vs ep32 实测=340us @tok256 w8a8）。这个高估让所有高并发
（bs>1）配置的 TPOT/TTFT 算超 SLO 被滤掉，寻优只剩 **bs=1**（档1 旧值 5.75
tok/s/gpu）。

补采 ep32 实测后，bs=16/并发128 的 TTFT 压到 1980<2000、TPOT 31<70 达标，单卡吞吐
跃至 **62 tok/s/gpu**。单变量 A/B 验证（移除 ep32 行干净退回 bs=1/5.75）确认这是
**ep32 数据单独导致**，非其他改动。**旧的"全 bs=1、5.75 tok/s/gpu"是 ep8-clamp
高估 dispatch 的 artifact，已作废。**

### 关键结论

1. **最优全是 agg（聚合）模式**：档1/档3 disagg 无可行解，档2 disagg 仅 ~0.47× agg。
   prefill 由 KV transfer 主导（1.7~2.6s），PD 分离把 KV 流式传输开销暴露在关键路径，
   反被 agg（无跨节点 KV transfer）超过。
2. **TPOT 全档 26~34ms，远达标（<70）**；TTFT 全档达标。推翻 05-30 版「TPOT 87ms
   超标」——decode 模型修正后准确（batch=1 对账 −0.6%）。
3. 单卡吞吐随 isl 升高而降（62→39→9 tok/s/gpu），长序列下达标所需的并发 batch 变小
   （档1 bs16 → 档3 bs2），prefill/KV 占比上升。

### 为何档3（isl=40k）单卡只有 9.2 —— KV 内存墙

不是吞吐模型异常，是**内存约束**：GLM-5 DSA 单 token 全层 KV ≈ 107 KB（kv_lora512 +
qk_rope64 + indexer128，×78 层 ×2B），单请求 KV 随 isl 线性涨：

| 档 | isl | 单请求 KV | 达标 batch | 单卡 tok/s |
|---|---|---|---|---|
| 1 | 10k | 1.1 GB | 16 | 62 |
| 2 | 20k | 2.2 GB | 11 | 39 |
| 3 | 40k | 4.4 GB | **2** | 9.2 |

910B 单卡 ~64GB HBM，扣 w8a8 权重（~10.5GB/卡）+ 激活 buffer 后留给 KV 的有限，KV
翻倍 → 能塞的 batch 大致减半，batch 从 16→11→2 断崖收缩，单卡吞吐随之跳水。验证：
**放宽 SLO 到 ttft=30s/tpot=200ms，档3 仍 bs≤2 / 9.2**，排除 SLO 因素，纯内存墙。
这是物理约束，非寻优可解——提高需加卡（更多 HBM）或 KV 量化（int8 KV 使单请求 KV
减半、batch 翻倍）。

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
