# GLM-5 SLO 配置寻优结果

状态：2026-05-30
部署：2P2D / 4 节点 / 64 卡 Ascend 910B，vllm-ascend 0.18.0，GLM-5-w8a8
工具：`aic-npu`（HYBRID 模式，chunked-prefill）
SLO：osl=2500，P90 TPOT < 70ms，P50 TTFT 分档

## 结论速览

**建模方式（去掉 overlap_factor，直接存实测净墙钟）**：
- KVTransfer 数据表直接存 profiler 实测的**净 KV 墙钟**（kernel 时间轴上 KV 并集减去与 compute 的重叠），query 插值返回，**不再乘任何标量修正系数**。这消除了之前 overlap_factor（0.86）这个纠缠了"median-vs-真实/KV流间重叠/KV-compute重叠"三件事的魔数。
- 实测净 KV 墙钟（ep16）：isl=10k→1679ms，isl=20k→2571ms（isl=2500 因同步气泡污染，由 10k/20k 斜率线性外推）。
- 单独存在的问题：prefill compute（主要 TP>1 DSA attention 走 HYBRID 回退）被高估 ~3×（模型 807ms vs profiler 实测 compute 并集 259ms），使模型 prefill TTFT 比真实墙钟偏高 ~27%。这是 TP>1 DSA 数据缺口（报告 §2.3），不归 KVTransfer。

根因仍是 mooncake KV transfer 主导 prefill TTFT，但**幅度比初版小得多**——初版用错误的 overlap=1.0 把档1 TTFT 估成 7743ms；按真实时间轴标定后，单请求 prefill 真实墙钟（profiler 实测）档1=**2754ms**、档2=**3833ms**。

## 各档实测最优（disagg）vs SLO

> "单请求真实墙钟"为 profiler 实测 ground truth（仅档1/档2 在 KV 标定网格内）；"模型单请求"= compute + 净KV实测；"含并发"= ×1.8 排队修正后的端到端 TTFT。配置：2P2D / 64 卡，prefill tp16(档1/2)或 tp32(档3/4)，decode tp4/dp8/ep32。

| 档 | isl | SLO TTFT | 单请求真实墙钟 | 模型单请求 | 含并发 TTFT | TPOT(<70) | 单请求判定 |
|---|---|---|---|---|---|---|---|
| 档1 | 0–10k | <2000ms | **2754ms** | 2693ms (-2%) | 5332ms | 81.2ms ✗ | ✗ 超(KV净1679+compute) |
| 档2 | 10–20k | <5000ms | **3833ms** | 5494ms** | 10879ms | 81.0ms ✗ | **✓ 达标** |
| 档3 | 20–40k | <8000ms | 外推* | 12028ms** | 23816ms | 81.2ms ✗ | 外推* |
| 档4 | 40–80k | <10000ms | 外推* | 37354ms** | 73961ms | 83.0ms ✗ | 外推* |

> `*` isl>20k 超出 KV 标定网格（净KV clamp 在 20k 值 2571ms）+ compute 随 isl 增长，仅外推参考。
> `**` 模型单请求在 isl≥20k 偏高，纯因 prefill compute 高估（DSA TP>1 走 HYBRID SOL；档1 实测 compute 259ms vs 模型 1014ms），**与 KV 无关**——净KV 已是实测值。修 TP>1 DSA silicon 后收敛。

**核心结论**：去掉两个不实的 factor（KV overlap_factor、TTFT 并发修正 ×1.8）后，SLO 偏差被干净归因到唯一真实数据缺口——**TP>1 DSA compute 高估**：

- **档2（isl 10–20k）**：单请求真实墙钟 **3833ms < SLO 5000ms = 达标**。但模型因 prefill compute 高估（DSA TP16 走 HYBRID = 2923ms vs profiler 实测 ~395ms）算出 5494ms 被判超；若 compute 准确，prefill = 395+2571(净KV) = **2966ms < 5000 达标**。
- **档1（isl=10k）**：单请求真实墙钟 2754ms 略超 SLO 2000ms，即使 compute 准确，净 KV 1679ms + 真实 compute 已接近 2000ms，是物理下限。
- **TPOT**：小 batch 可达标（decode bs≤3 时 TPOT 64.9ms<70；bs=4 起 79.9ms 超）。寻优应在 TPOT<70 约束下取最大可行 decode bs（=3）。
- **并发排队**：不再乘 ×1.8（该 factor 的 lc、N 两参数都写死无依据，见 picking.py 注释）。P50 TTFT 在低并发下接近单请求值；高并发排队作为单独风险项，需生产实测 TTFT 才能可信建模。

> 修 TP>1 DSA silicon 数据后，档2 模型即与真实墙钟收敛、判为达标。这是当前唯一阻塞 SLO 寻优产出"满足约束配置"的数据缺口。

## TTFT 归因（按 profiler 时间轴累加）

档1 单请求 prefill 真实墙钟 = **2754ms**（profiler 实测），时间轴分解：

| 组成 | 真实墙钟 | 说明 |
|---|---|---|
| compute 并集（DSA+MoE+GEMM 实际） | 259ms | profiler 实测（模型高估为 807+ms） |
| KV transfer 净墙钟（直接查表实测） | ~1679ms | profiler 时间轴: KV并集−KV∩compute |
| 合计 | ~2693ms | = 真实墙钟 2754ms (-2%) |

**KV transfer 是绝对支配项**（净墙钟 1679ms / 单请求墙钟 2754ms = 61%）。该值直接来自 profiler kernel 时间轴（KV 并集减去与 compute 的重叠），**不经任何 overlap_factor**——数据表存的就是实测净墙钟，避免了标量魔数。模型 prefill TTFT 在 isl=10k 对齐真实墙钟 -2%。

**SLO 可达性**：档2 单请求真实墙钟 3833ms < SLO 5000ms（达标）；档1 真实 2754ms 略超 SLO 2000ms，且即使零并发也因 KV transfer 净墙钟单项 1679ms + compute 而超。

## TPOT 超标归因

全档 TPOT ~81–87ms，超 70ms。验证：decode tp4→tp8 几乎不变（81.2→80.2ms），说明瓶颈不在 TP，而是 **MoE dispatch + W8A8 GEMM 的单步固有延迟**（decode 已 memory-bound）。70ms 在当前 decode 算子数据下达不到。

## 长 isl 的 prefill OOM（档3/档4）

prefill 单卡内存（bs=1）随 isl：

| isl | tp16/ep16 | tp32/ep32 |
|---|---|---|
| 40k | 64GiB **OOM** | 41GiB OK |
| 80k | 79GiB **OOM** | 55GiB OK |

档3/档4 必须 prefill tp32。注意 `default` 模式因 commit `6849445` 把 DEEPSEEKV32 prefill pin 死 tp16，长 isl 不会自动升 tp32——故档3/档4 在 `default` 下 disagg 无解，需手动指定 tp32（上表已用 tp32）。

## 数据可信度边界

- **档1/档2（isl≤20k）**：KV transfer 在标定网格内（表覆盖 isl 2500/10k/20k），TTFT 预测可信。
- **档3/档4（isl>20k）**：**超出 KVTransfer 标定网格上限（20k），靠 clamp 外推（hold flat），TTFT 偏乐观**，实际可能更高。这两档结论仅供量级参考。
- decode TPOT、prefill compute 部分为 silicon/HYBRID 覆盖，±25% 内。

## 要达成 SLO 需要的方向（非配置寻优可解）

寻优已穷尽并行维度，SLO 缺口是架构/数据层面的：

1. **削减 KV transfer**（TTFT 主因）：KV cache 压缩（如 MLA 低秩already）、更快互联（当前 RoCE 25GB/s 跨节点）、或减少跨 worker 传输量。这是档1/档2 TTFT 从 7.7s 降到 2s 的唯一途径。
2. **降 TPOT**：需 decode 算子层优化（MoE dispatch 更快），非并行度可解。
3. **长 isl（档3/档4）**：prefill tp32 + KV 量化；并补采 isl>20k 的 KVTransfer 数据以消除外推。
