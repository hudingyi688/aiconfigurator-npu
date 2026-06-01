# GLM-5 SLO 配置寻优结果

状态：2026-05-30
部署：2P2D / 4 节点 / 64 卡 Ascend 910B，vllm-ascend 0.18.0，GLM-5-w8a8
工具：`aic-npu`（HYBRID 模式，chunked-prefill）
SLO：osl=2500，P90 TPOT < 70ms，P50 TTFT 分档

## 结论速览

**校准修正后**（overlap_factor 1.0→0.60，依据 profiler 真实单请求 step 墙钟，见附注）：
- **档2（isl 10–20k）单请求 prefill 3833ms < SLO 5000ms，达标**；档1（isl=10k）单请求 2754ms 仍略超 SLO 2000ms。
- 但叠加并发排队（×1.8）后档1/档2 端到端 TTFT 超 SLO；TPOT 全档 ~81–87ms 超 70ms。
- 长 isl（档3/档4）受 prefill OOM + KV transfer 主导，差距更大。

根因仍是 mooncake KV transfer 主导 prefill TTFT，但**幅度比初版报告小得多**——初版用错误的 overlap=1.0 把档1 TTFT 估成 7743ms，校准到真实墙钟后是 5447ms（单请求 2754ms）。

## 各档实测最优（disagg）vs SLO（校准后）

| 档 | isl | SLO TTFT | 单请求 prefill 墙钟 | 含并发(×1.8) TTFT | 差距 | TPOT(SLO<70) | prefill | decode |
|---|---|---|---|---|---|---|---|---|
| 档1 | 0–10k | <2000ms | **2754ms** | ~5447ms | 1.4×(单)/2.7×(并发) | 85.5ms ✗ | tp16/ep16 | tp4/dp8/ep32 |
| 档2 | 10–20k | <5000ms | **3833ms ✓单请求达标** | ~10539ms | 0.8×(单)/2.1×(并发) | 86.7ms ✗ | tp16/ep16 | tp4/dp8/ep32 |
| 档3 | 20–40k | <8000ms | ~6000ms* | OOM→tp32 | — | 81.2ms ✗ | tp32/ep32 | tp4/dp8/ep32 |
| 档4 | 40–80k | <10000ms | 偏保守* | tp32 | — | 83.0ms ✗ | tp32/ep32 | tp4/dp8/ep32 |

> 单请求墙钟已对齐 profiler 实测（档1 -0%，档2 偏差小）。`*` 标记 isl>20k 超出 KVTransfer 标定网格，clamp 外推偏保守。"含并发"列为 ×1.8 排队修正后，反映高并发场景。

## TTFT 归因（校准后，profiler 真实墙钟）

档1 单请求 prefill 真实墙钟 = **2754ms**（profiler step_trace + kernel 时间轴跨度，两口径吻合）：

| 组成 | 延迟 | 说明 |
|---|---|---|
| compute（DSA attention + MoE + GEMM） | ~1014ms | silicon/HYBRID |
| **KVTransfer**（device_total 2896ms × overlap 0.60） | **~1738ms** | 校准后净墙钟贡献 |
| 单请求 prefill 合计 | ~2752ms | 对齐真实 2754ms |

**关键修正**：初版报告称"KV transfer 与 compute 重叠仅 8–10%、overlap=1.0"是**错误**的——那个标定只量了 KV-kernel 之间的并集，漏了 KV kernel 与 compute 跨流并行。profiler 实测：单请求 prefill 所有 kernel duration 之和 6828ms ≫ 真实墙钟 2754ms，证明大量跨流重叠。校准到真实墙钟后 overlap_factor=0.60（详见 `perf_database.py:_KV_TRANSFER_OVERLAP_FACTOR` 注释）。

**SLO 可达性重判**：档2 单请求其实达标（3833<5000）；档1 单请求略超（2754 vs 2000），即使零并发也因 KV transfer 单项 1738ms 而超。是否最终达标取决于真实并发排队倍数（生产 bench TTFT 未保存，无法精确，×1.8 是经验上界）。

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
