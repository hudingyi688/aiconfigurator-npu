# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIConfigurator-NPU: Performance database and configuration search for Ascend NPU."""

__version__ = "0.1.0"

from aiconfigurator_npu.sdk import common
from aiconfigurator_npu.sdk.task import TaskConfig, TaskRunner
from aiconfigurator_npu.sdk.perf_database import get_database, set_systems_paths

__all__ = [
    "__version__",
    "common",
    "TaskConfig",
    "TaskRunner",
    "get_database",
    "set_systems_paths",
]