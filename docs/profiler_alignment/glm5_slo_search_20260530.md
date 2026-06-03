# GLM-5 SLO 配置寻优结果

状态：2026-05-30
部署：2P2D / 4 节点 / 64 卡 Ascend 910B，vllm-ascend 0.18.0，GLM-5-w8a8
工具：`aic-npu`（HYBRID 模式，chunked-prefill）
SLO：osl=2500，P90 TPOT < 70ms，P50 TTFT 分档

## 结论速览

DSA prefill 已改为**实测 chunked 数据**（NPU 采集 w8a8/num_heads=4，按 ceil(isl/4096) 个 prefill step 累加），取代之前 full-seq 高估或 HYBRID SOL 估算。基于此重跑 4 档：

- **4 档 prefill TTFT 全部超 SLO**；TPOT 全档 ~87ms 超 70ms。
- **关键认知更新**：chunked 修正后 **DSA（稀疏注意力，随 isl 线性增长的 per-step 重算）成为 prefill 主导项**，超过 KV transfer——推翻了之前"KV transfer 占 prefill 87%"的旧结论（那基于 full-seq 错误数据）。

## 各档实测对比（disagg prefill，2P2D / 64 卡）

> 数据：HYBRID 模式，prefill 各档真实配置（tp16/tp32），decode tp4/dp8/ep32。TTFT 为单请求 prefill（未叠加并发）。

| 档 | isl | prefill 配置 | num_heads | SLO TTFT | 实测 prefill TTFT | 差异 | DSA | KV transfer | other | 可信度 |
|---|---|---|---|---|---|---|---|---|---|---|
| 档1 | 0–10k | tp16/ep16 | 4 (命中) | <2000ms | **5201ms** | **2.6× 超** | 3315ms | 1679ms | 207ms | 🟢 可信 |
| 档2 | 10–20k | tp16/ep16 | 4 (命中) | <5000ms | **9024ms** | **1.8× 超** | 6153ms | 2571ms | 300ms | 🟢 可信 |
| 档3 | 20–40k | tp32/ep32 | 2 (外推) | <8000ms | 40419ms | 5.1× 超 | 37364ms | 2571ms | 484ms | 🔴 不可信 |
| 档4 | 40–80k | tp32/ep32 | 2 (外推) | <10000ms | 263416ms | 26× 超 | 259989ms | 2571ms | 856ms | 🔴 不可信 |

> 可信度说明：
> - 🟢 **档1/档2**：prefill tp16 → num_heads=4，**精确命中** silicon 表采集点，DSA 数据可信。
> - 🔴 **档3/档4**：prefill tp16 在 isl≥40k **OOM**（KV cache 装不下），只能用 tp32；但 tp32 → num_heads=2 **落在 silicon 表 {4,64} 范围外**，DSA 外推失真（260s 是 artifact，非真实）。

## TPOT（decode 侧）

| decode 配置 | batch | TPOT | SLO<70 |
|---|---|---|---|
| tp4/dp8/ep32 | bs=3 | 87ms | ✗ 超 |

decode TPOT 主要由 MoE dispatch + GEMM 单步固有延迟决定，降 batch 也压不到 70ms 以下（isl 越大 KV 越多，TPOT 略升）。

## 各档 GAP 分析

| 档 | 主要 GAP | 量化 | 性质 |
|---|---|---|---|
| 档1 | DSA(3.3s) + KV(1.7s) 已超 SLO 2s | prefill 5.2s vs 2s | DSA 主导，物理瓶颈 |
| 档2 | DSA(6.2s) 单项就超 SLO 5s | prefill 9s vs 5s | DSA 主导 |
| 档3/档4 | 数据缺口（num_heads=2 未采）+ tp16 OOM | 无法可信评估 | 数据/内存模型缺口 |
| 全档 | TPOT 87ms > 70ms | decode 算子固有 | 非并行可解 |

**核心 GAP**：chunked 修正后 **DSA 是 prefill 的真实主导项**（档1 占 64%、档2 占 68%），且随 isl 线性增长（每 step 重算累积 KV 的 sparse attention）。降 prefill TTFT 的重点从"减 KV transfer"转向"减 DSA per-step 重算"。

> ⚠️ 遗留验证：DSA 累加绝对值（档1 3.3s）的单算子实测可信，但**完整 prefill 累加结果缺单请求 profiler 端到端验证**（profiler 仅采样窗口）。相对趋势（DSA 随 isl 线性、超 KV）是 chunked 物理特性，方向可信。

## 下一步计划

按优先级：

1. **补采 num_heads=2（tp32）DSA 数据** —— 解锁档3/档4 的可信评估（当前 tp16 OOM、tp32 外推失真两头堵）。命令同前，加 `--num-heads-override 2`。
2. **单请求 profiler 端到端验证**（遗留问题）—— 跑一个不并发的 isl=10k prefill profiler，数真实 SFA 总次数 + 完整 prefill ttft，锚定 DSA 累加公式（3.3s）和 KV transfer 的绝对值。这是消除所有"绝对值待验证"标注的唯一硬锚点。
3. **SLO 可达性结论**（数据补全后）—— 当前档1/档2 可信地超 SLO，主因 DSA。若要达标需架构层优化（更激进稀疏 / 减 chunk 重算 / 降 TPOT 的 decode 算子优化），非配置寻优可解。
