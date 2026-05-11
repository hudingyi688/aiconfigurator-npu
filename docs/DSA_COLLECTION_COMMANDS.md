# DSA Module 采集命令

> **注意**：
> 1. 需要在项目根目录运行，并设置 `PYTHONPATH=collector`
> 2. 已创建本地 GLM-5 配置文件 `model_configs/zai-org--GLM-5_config.json`，无需访问 HuggingFace Hub

## 快速验证（单点测试）

```bash
PYTHONPATH=collector python collector/npu/collect_mla_module.py --mode context --quick --batch-size 4 --seq-len 2048 --output-dir ./data/glm5_dsa_module
```

## 完整采集

### Context 模式

```bash
PYTHONPATH=collector python collector/npu/collect_mla_module.py --mode context --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
```

### Generation 模式

```bash
PYTHONPATH=collector python collector/npu/collect_mla_module.py --mode generation --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
```

### 一键采集（Context + Generation）

```bash
PYTHONPATH=collector python collector/npu/collect_mla_module.py --mode context --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module && PYTHONPATH=collector python collector/npu/collect_mla_module.py --mode generation --model zai-org/GLM-5 --output-dir ./data/glm5_dsa_module
```

## 复制到数据目录

```bash
cp data/glm5_dsa_module/dsa_context_module_perf.txt systems/data/ascend_910b/vllm-ascend/0.18.0/ && cp data/glm5_dsa_module/dsa_generation_module_perf.txt systems/data/ascend_910b/vllm-ascend/0.18.0/
```