# GLM-5 SLO 配置寻优结果

# GLM-5 SLO 配置寻优结果

> ⚠️ **已过时 (2026-06-04)。本文档的「DSA 是 prefill 绝对主导 (64%→92%)」结论已作废**
> ——它基于错误的 DSA 口径（nh=4 合成 silicon + query 未按 Context Parallelism 切分，
> 把单请求 DSA 放大了约 20×）。修正后 prefill 由 KV transfer 主导，DSA 仅占 8%~23%。
> 另外「TPOT 高估 1.6×」也已作废（错误对账所致，模型 TPOT 实际 −0.6% 准确）。
> **请改用 [glm5_slo_search_20260604.md](./glm5_slo_search_20260604.md)。** 以下内容仅作历史存档。

---

状态：2026-05-30
部署：2P2D / 4 节点 / 64 卡 Ascend 910B，vllm-ascend 0.18.0，GLM-5-w8a8
工具：`aic-npu`（HYBRID 模式，chunked-prefill）
SLO：osl=2500，P90 TPOT < 70ms，P50 TTFT 分档

## 结论速览

DSA prefill 已改为**实测 chunked 数据**（NPU 采集 w8a8，num_heads=4/2 对应 tp16/tp32，按 ceil(isl/4096) 个 prefill step 累加），取代之前 full-seq 高估或 HYBRID SOL 估算。基于此重跑 4 档：

- **4 档 prefill TTFT 全部超 SLO**（超 1.8×~4.4×）；TPOT 全档 ~87ms 超 70ms。
- **关键认知更新**：chunked 修正后 **DSA（稀疏注意力，随 isl 线性增长的 per-step 重算）成为 prefill 主导项**（占 64%~92%），超过 KV transfer——推翻了之前"KV transfer 占 prefill 87%"的旧结论（那基于 full-seq 错误数据）。

## 各档实测对比（disagg prefill，2P2D / 64 卡）

> 数据：HYBRID 模式，prefill 各档真实配置（tp16/tp32），decode tp4/dp8/ep32。TTFT 为单请求 prefill（未叠加并发）。

| 档 | isl | prefill 配置 | num_heads | SLO TTFT | 实测 prefill TTFT | 差异 | DSA | KV transfer | other | 可信度 |
|---|---|---|---|---|---|---|---|---|---|---|
| 档1 | 0–10k | tp16/ep16 | 4 (命中) | <2000ms | **5201ms** | **2.6× 超** | 3315ms | 1679ms | 207ms | 🟢 可信 |
| 档2 | 10–20k | tp16/ep16 | 4 (命中) | <5000ms | **9024ms** | **1.8× 超** | 6153ms | 2571ms | 300ms | 🟢 可信 |
| 档3 | 20–40k | tp32/ep32 | 2 (命中) | <8000ms | **18020ms** | **2.3× 超** | 14965ms | 2571ms | 484ms | 🟢 可信 |
| 档4 | 40–80k | tp32/ep32 | 2 (命中) | <10000ms | **44382ms** | **4.4× 超** | 40955ms | 2571ms | 856ms | 🟢 可信 |

> 全 4 档 num_heads（tp16→4、tp32→2）均**精确命中** silicon 采集点，DSA 数据可信。补采 num_heads=2 后，档3/档4 从之前外推 artifact（40s/260s）修正为真实值（18s/44s）。注：num_heads=2 与 4 延迟接近（DSA module 在小头数区由投影 GEMM/量化/KV 访存主导，非头数本身），这正是之前外推到 {4,64} 区间外失真的原因。

## TPOT（decode 侧）

| decode 配置 | batch | TPOT | SLO<70 |
|---|---|---|---|
| tp4/dp8/ep32 | bs=3 | 87ms | ✗ 超 |

decode TPOT 主要由 MoE dispatch + GEMM 单步固有延迟决定，降 batch 也压不到 70ms 以下（isl 越大 KV 越多，TPOT 略升）。

### decode TPOT 的 profiler 校验（batch=1 干净对账）

用生产单请求 decode profiler（isl=10k，batch=1，num_heads=16）整体对账，发现模型 **TPOT 高估 ~1.6×**：

| | 模型 batch=1 | profiler batch=1 实测 |
|---|---|---|
| 单步 TPOT | 52.7ms | **32.7ms** |
| 构成 | MoE 27.6 + attention 24.3（串行相加）| compute 并集 29.1 + 通信 1.5 |

**根因：模型把各 module（attention、MoE）silicon 值纯串行累加，但生产硬件在 module 间 / 层间多流并行**——profiler kernel 总和 76.3ms 被并行压缩到 30.6ms 并集（2.5× 压缩）。模型缺这个跨 module 并行折扣。

> 校验澄清的几点：(1) 各 module 的 silicon 单值是准的（generation DSA 单层 silicon 310us vs profiler DSA module 184~301us，比值 1.0~1.7×，在 module 边界口径差异内，**非数量级高估**）；(2) 之前怀疑的 "KV pool batch_get 70ms 主导 TPOT" 不成立——batch=1 run 根本无 batch_get 也能正常跑，它非必需主导项，且两 profiler（batch 1 vs 7）batch 不可比，无法隔离其贡献；(3) TPOT 真实缺口是 **module 间并行未建模**，需实测驱动的并行折扣（kernel 和 vs 并集 ≈2.5×，待多 batch/配置验证），不是单个算子或 KV pool。

## 各档 GAP 分析

| 档 | 实测 prefill | SLO | 超出 | 主因 |
|---|---|---|---|---|
| 档1 | 5.2s | 2s | 3.2s | DSA 3.3s 主导（占 64%）|
| 档2 | 9.0s | 5s | 4.0s | DSA 6.2s 单项就超 SLO（占 68%）|
| 档3 | 18.0s | 8s | 10.0s | DSA 15s 主导（占 83%）|
| 档4 | 44.4s | 10s | 34.4s | DSA 41s 主导（占 92%）|
| 全档 | TPOT 87ms | 70ms | 17ms | decode 算子固有，非并行可解 |

**核心 GAP**：chunked 修正后 **DSA 是 prefill 的绝对主导项**，占比随 isl 升高（64%→92%），且 DSA 随 isl 近线性增长（每 step 重算累积 KV 的 sparse attention）。降 prefill TTFT 的重点从"减 KV transfer"转向**"减 DSA per-step 重算"**。

> ⚠️ 遗留验证：DSA 单 step、单算子延迟为真机实测（可信）；但**完整 prefill 的累加结果缺单请求 profiler 端到端验证**（现有 profiler 仅采样窗口）。各档相对趋势（DSA 主导、随 isl 增长）是 chunked 物理特性，方向可信；绝对值待端到端锚定。

## 下一步计划

1. **单请求 profiler 端到端验证**（遗留问题）—— 跑一个不并发的 isl=10k 完整 prefill profiler，数真实 SFA 总次数 + 完整 prefill TTFT，锚定 DSA 累加公式与 KV transfer 的绝对值。这是把 4 档从"趋势可信"升级到"数值可信"的唯一硬锚点。
2. **SLO 达标方向** —— 4 档均可信地超 SLO，主因 DSA（占 64%~92%）。达标需架构层优化：更激进稀疏 / 减 chunk 重算（DSA）+ decode 算子优化（TPOT），**非配置寻优可解**。
3. **配置形态结论** —— 当前数据支撑「相对排名 / 配置形态选择」可信用途；绝对 TTFT 达标判定待第 1 项验证。
