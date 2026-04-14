# AIConfigurator for NPU

Operator microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Adapted from [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) collector design, targeting Ascend NPU with vllm-ascend framework alignment.

## Structure

```
collector/
  bench_engine.py          # NPU Event timing engine with NPU Graph capture + replay
  npu/
    gemm_factory.py        # GEMM operator factory (BF16 / W8A8_DYNAMIC)
    collect_gemm.py        # GEMM microbenchmark collector
    attn_factory.py        # Attention operator factory (Context / Decode)
    collect_attn.py        # Attention microbenchmark collector
    moe_factory.py         # MoE operator factory (BF16 / W8A8_DYNAMIC)
    collect_moe.py         # MoE microbenchmark collector
```

## Features

- NPU Graph capture + replay to eliminate Python dispatch overhead (~30-40us → ~10us/op)
- Real kernel path via vllm-ascend framework layer (not raw torch_npu calls)
- FRACTAL_NZ weight format support for W8A8 quantized models
- 6-op L2 cache flush rotation (GEMM) aligned with AIConfigurator GPU version
- CSV output compatible with TensorCast profiling database format
- Checkpoint/resume support for large parameter sweeps

## Requirements

- Ascend NPU with CANN 8.5+
- vLLM 0.18.0+ with vllm-ascend
- torch-npu

## Quick Start

```bash
# GEMM
python collector/npu/collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./gemm_data

# Attention
python collector/npu/collect_attn.py --op-types context generation --output-dir ./attn_data

# MoE
python collector/npu/collect_moe.py --quant-types bf16 w8a8_dynamic --output-dir ./moe_data
```
