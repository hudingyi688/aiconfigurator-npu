#!/usr/bin/env python3
"""Check vllm/vllm-ascend version compatibility for DSA module collection.

Run this on NPU before executing collect_mla_module.py to identify
missing modules/API changes early.
"""

import sys

checks = [
    ("torch_npu", "import torch_npu"),
    ("vllm_ascend", "import vllm_ascend"),
    ("vllm_ascend.ops.mla", "from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention"),
    ("vllm_ascend.ops.mla", "from vllm_ascend.ops.mla import IndexerWrapper"),
    ("vllm_ascend.ascend_forward_context", "from vllm_ascend.ascend_forward_context import set_ascend_forward_context"),
    ("vllm_ascend.utils", "from vllm_ascend.utils import set_weight_prefetch_method"),
    ("vllm_ascend.ascend_config", "from vllm_ascend.ascend_config import WeightPrefetchConfig"),
    ("vllm.model_executor.layers.mla", "from vllm.model_executor.layers.mla import MLAModules"),
    ("vllm.model_executor.layers.linear", "from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear"),
    ("vllm.model_executor.layers.layernorm", "from vllm.model_executor.layers.layernorm import RMSNorm"),
    ("vllm.model_executor.layers.rotary_embedding", "from vllm.model_executor.layers.rotary_embedding import get_rope"),
    ("vllm.config", "from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig"),
    ("vllm_ascend.attention.sfa_v1", "from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata"),
]

def main():
    print("Checking vllm/vllm-ascend compatibility...\n")
    
    passed = 0
    failed = []
    
    for module, stmt in checks:
        try:
            exec(stmt)
            print(f"  [OK] {module}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            failed.append((module, str(e)))
    
    print(f"\n{passed}/{len(checks)} checks passed")
    
    if failed:
        print("\nMissing/incompatible modules:")
        for module, err in failed:
            print(f"  - {module}: {err}")
        print("\nPlease upgrade vllm-ascend or check import paths.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Ready for DSA collection.")
        sys.exit(0)

if __name__ == "__main__":
    main()