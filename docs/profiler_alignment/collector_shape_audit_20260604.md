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

2. **moe_dispatch_combine 缺生产 ep32（待采）**：现仅 ep2/4/8。生产 ep32 靠 clamp 到 ep8 近似，偏差未量化。采集命令见下。

## agent 误报、实测澄清（不是问题）

- DSA **有 w8a8**（context nh2/4、generation nh2/16 均 w8a8_dynamic）——非"全 float16"。
- MoE GLM-5 **复用 deepseek-v3 行**（7168/2048/256/8 同维度）是设计，非缺失。
- `gemm_perf` / `moe_perf` / `dsa_context_attn_core` 两树**已同步**。
- `collect_attn`（标准 MHA）对 GLM-5 **不适用**（走 DSA/SFA 路径），非缺口。

## 待办采集命令

### ep32 moe_dispatch_combine（生产 EP）
```bash
# NPU，需 32 卡 torchrun（或按 collector 的 world_size 约定）
cd collector/npu
torchrun --nproc_per_node=8 --nnodes=4 collect_moe_dispatch_combine.py \
  --ep-size 32 --hidden 6144 --inter 2048 --num-experts 256 --topk 8 \
  --quant-types bf16 w8a8_dynamic \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --output-dir ./moe_dispatch_ep32
```
采回后合并进 `moe_dispatch_combine_perf.txt`（两树都要更新），解锁生产 ep32 精确值（当前 clamp 到 ep8）。
