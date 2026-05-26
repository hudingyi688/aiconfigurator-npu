# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal generator stub.

The full upstream `aiconfigurator.generator` package is responsible for
turning a chosen config into runnable artifacts (vllm / trtllm /
sglang start scripts, k8s yaml, dynamo deployment, ...). The NPU port
focuses on the *search* path — picking the best (TP, EP, ...) for a
given SLA and printing the Pareto table — and does not need backend
artifact generation.

This stub provides every name imported from `aiconfigurator_npu.generator.api`
and `aiconfigurator_npu.generator.module_bridge` so the CLI can boot.
Any call site that actually exercises generation is either:
  - guarded by a try/except (e.g. report_and_save.generate_backend_artifacts),
    so it logs a warning and continues; or
  - only reachable from `aic-npu generate`, which is not supported on NPU.

If full generator support is needed later, port
upstream `src/aiconfigurator/generator/` into this package.
"""
