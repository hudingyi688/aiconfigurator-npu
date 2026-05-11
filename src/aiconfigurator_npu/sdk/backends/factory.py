# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator_npu.sdk import common
from aiconfigurator_npu.sdk.backends.base_backend import BaseBackend
from aiconfigurator_npu.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator_npu.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator_npu.sdk.backends.vllm_backend import VLLMBackend


def get_backend(backend_name: str) -> BaseBackend:
    """
    Get the backend class by the backend name.

    Raises:
        ValueError: If the backend name is not found.
    """
    backend_map = {
        common.BackendName.trtllm: TRTLLMBackend,
        common.BackendName.sglang: SGLANGBackend,
        common.BackendName.vllm: VLLMBackend,
        common.BackendName.vllm_ascend: VLLMBackend,
    }

    try:
        backend_enum = common.BackendName(backend_name)
    except ValueError:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend_class = backend_map.get(backend_enum)
    if backend_class is None:
        raise ValueError(f"Unknown backend: {backend_name}")

    return backend_class()
