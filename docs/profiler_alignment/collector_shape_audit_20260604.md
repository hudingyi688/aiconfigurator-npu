# GLM-5 采集 shape 覆盖检查报告

状态：2026-06-04
范围：`collector/npu/` 下各采集脚本的 shape 覆盖 vs GLM-5 生产配置，以及已采数据文件的实际覆盖。
背景：多次会话中分片补采（nh=2/4/16/64、w8a8、chunked-prefill），需核查覆盖缺口与不一致。

## GLM-5 生产配置基线

- 78 层，hidden=6144，num_attention_heads=64，MoE 256 experts / topk=8 / moe_inter=2048（MoE 输入 hidden=7168，同 DeepSeek-V3）。
- DSA：q_lora=2048，kv_lora=512，qk_nope=192，qk_rope=64，v_head=256，index_topk=2048，index_n_heads=32。
- prefill tp16/dp2/ep32（DSA = Context Parallel，每卡全 64 头，per-rank query=256）。
- decode tp4/dp8/ep32（DSA 头维切分，tp4→16 头）；长档 tp32→2 头。
- max_num_batched_tokens=4096，w8a8（gemm_type=w8a8_dynamic），MTP=2。

## 实测覆盖（运行时加载的 package 树 `src/aiconfigurator_npu/systems/`）

| 算子表 | 实际覆盖 | 生产需求 | 结论 |
|---|---|---|---|
| `dsa_context_module` | nh2/4 w8a8(122行) + nh64 float16(124行) | prefill nh64 | nh64 仅 float16，但 **prefill 已改走 profiler-derived 表**，此表不再用于 GLM-5 prefill |
| `dsa_context_attn_core`（新）| nh64 CP16 q256，KV{4096..20480} | prefill nh64 CP | ✓ profiler 实测，prefill 实际数据源 |
| `dsa_generation_module` | nh2/16 w8a8 + nh64 float16(333行) | decode nh16(tp4)/nh2(tp32) | ✓ 生产 nh16/nh2 均 w8a8 命中 |
| `moe_dispatch_combine` | ep2/4/8 × {bf16,w8a8} | **ep32** | ⚠️ **缺 ep32**，现 clamp 到 ep8 近似 |
| `gemm_perf` / `moe_perf` | 见下 | — | GLM-5 MoE 复用 deepseek-v3 行（同维度），设计如此 |

## 真实问题（已处理 / 待办）

1. **两 systems 树不同步（已修，2026-06-04）**：
   - 角色：root `systems/` = 转换 input-dir（原始 CSV + 转换产物 txt）；package `src/aiconfigurator_npu/systems/` = output-dir，**运行时只加载 package 树**（`perf_database._SYSTEMS_PATHS` 默认指向它）。
   - 问题：root 树的 DSA txt 停留在旧版（nh64/float16，124/333 行），且缺 `moe_dispatch_combine`；package 树是补采后的完整数据。运行时用 package 树所以结果无误，但 root 树是过时孤儿，曾误导分析（一度把 root 的 124 行旧表当作当前数据）。
   - 处理：把 `dsa_context_module` / `dsa_generation_module` / `moe_dispatch_combine` 从 package 树同步到 root 树，两树 txt 现一致。root 树 CSV（采集源）保留。

2. **moe_dispatch_combine 缺生产 ep32（已采，2026-06-04）**：原仅 ep2/4/8，生产 ep32 clamp 到 ep8（高估 dispatch ~16%）。已在 2×A3 节点补采 ep32 实测合并进两树，grid 现 {2,4,8,32}。修正后 SLO 单卡吞吐 +10.8×（见 glm5_slo_search_20260604.md）。

3. **architecture dims 错配（已知，标注不修 — YAGNI）**：GLM-5 config 声明 `DeepseekV32ForCausalLM`，但真实 dims（hidden6144/q_lora2048/v256/idx_n32）匹配 `GlmMoeDsaForCausalLM` 条目，非 DeepseekV32（hidden7168/q_lora1536）。`DeepSeekV32Model` 同时服务 GLM-5 和 DeepSeek-V3.2，靠 arch 字符串区分，但 `DSA_MODEL_DIMS` 按 arch 查 dims → GLM-5 取到 V3.2 的 dims。**影响面**：仅 `query_context_dsa_projection_sol`（GLM-5 prefill 投影 GEMM SOL），**低估 ~34%**（115 vs 174 us/层 @q256 nh64 w8a8），E2E 占 prefill 仅 **~0.69%**（投影是小头，attention 核心是 profiler 实测、不受影响）。silicon 表行标 DeepseekV32 且 silicon 命中不走 dims 公式，故 decode/silicon 路径不受影响。正确修复需把 config 实际 dims 穿进 op 管道或拆分共享 arch key，性价比低，故仅在代码（perf_database.py `query_context_dsa_projection_sol`）加注释标注，不改逻辑。

## agent 误报、实测澄清（不是问题）

- DSA **有 w8a8**（context nh2/4、generation nh2/16 均 w8a8_dynamic）——非"全 float16"。
- MoE GLM-5 **复用 deepseek-v3 行**（7168/2048/256/8 同维度）是设计，非缺失。
- `gemm_perf` / `moe_perf` / `dsa_context_attn_core` 两树**已同步**。
- `collect_attn`（标准 MHA）对 GLM-5 **不适用**（走 DSA/SFA 路径），非缺口。

## 待办采集命令

### moe_dispatch_combine 大 ep（生产 EP=32）

**A3 环境（单节点 16 DIE = 16 NPU device）**：EP 组是真实 HCCL 子组，需 `WORLD % ep == 0` 且每 rank 是物理 device，不能在更少 device 上模拟更大 ep。所以：
- **ep16 → 单节点（WORLD=16），现在就能采** —— 比当前 ep8 clamp 更接近生产 ep32，强烈建议先采这个。
- **ep32 → 2 个 A3 节点（2×16=32）** —— 不是 4 节点（那是按每节点 8 device 算的旧假设）。

封装脚本 `collector/npu/collect_moe_dispatch_ep32.sh`（自动探测 device_count，默认 EP=16/单节点）：

```bash
# 立即可做：单 A3 节点采 ep16
EP=16 NNODES=1 bash collector/npu/collect_moe_dispatch_ep32.sh

# 拿到第 2 个 A3 节点后：采 ep32（每节点都跑，NODE_RANK 0/1，MASTER_ADDR 同一个）
EP=32 NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash collector/npu/collect_moe_dispatch_ep32.sh  # 节点0
EP=32 NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash collector/npu/collect_moe_dispatch_ep32.sh  # 节点1
```
（`GPUS` 留空会自动探测 `torch.npu.device_count()`；A3 应为 16。若探测为 8，则每节点 8 device，ep32 需 4 节点。）

采回后合并 `moe_dispatch_combine_ep{16,32}.csv` 进**两个 systems 树**的 `moe_dispatch_combine_perf.txt`，ep16/ep32 查询即命中实测（当前 clamp 到 ep8）。
