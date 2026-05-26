# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stub for aiconfigurator_npu.generator.module_bridge.

See generator/__init__.py for rationale.
"""

from __future__ import annotations

from typing import Any


def task_config_to_generator_config(
    *,
    task_config: Any,
    result_df: Any = None,
    generator_overrides: Any = None,
) -> dict[str, Any]:
    """Return a placeholder generator config dict.

    The dict is later yaml-dumped to top{i}/generator_config.yaml; we
    emit enough fields that the dump succeeds and the followup
    generate_backend_artifacts NotImplementedError gets caught.
    """
    return {
        "model": getattr(task_config, "model_name", None) or
                 getattr(task_config, "model", None),
        "backend": getattr(task_config, "backend_name", None) or
                   getattr(task_config, "backend", None),
        "backend_version": getattr(task_config, "backend_version", None),
        "system": getattr(task_config, "system", None),
        "_note": (
            "aiconfigurator-npu generator is a stub; rerun against "
            "upstream aiconfigurator if you need backend artifacts."
        ),
    }
