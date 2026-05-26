# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stub for aiconfigurator_npu.generator.api.

See generator/__init__.py for rationale. None of these stubs implement
the upstream behaviour; they only ensure imports resolve and CLI
codepaths that don't strictly require generation can run.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_GENERATOR_DISABLED_MSG = (
    "Backend artifact generation is disabled in aiconfigurator-npu "
    "(generator/ is a stub). Use aiconfigurator-npu only for "
    "search and Pareto analysis."
)


def add_generator_override_arguments(parser) -> None:
    """No-op: upstream adds --generator-* flags. We accept none."""
    return


def generator_cli_helper(argv) -> bool:
    """No-op: upstream pre-CLI hook. Returns False so main() runs normally."""
    return False


def generate_naive_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """`aic-npu generate` is unsupported. Caller should not hit this."""
    raise NotImplementedError(_GENERATOR_DISABLED_MSG)


def generate_backend_artifacts(*args: Any, **kwargs: Any) -> None:
    """Called from default-mode result writeout, guarded by try/except.

    Raise so the caller's `except Exception` logs a warning and the
    Pareto run still completes. The .yaml file with the chosen config
    is already written before this is invoked.
    """
    raise NotImplementedError(_GENERATOR_DISABLED_MSG)


def get_default_dynamo_version_mapping() -> dict[str, str]:
    return {}


def load_generator_overrides_from_args(args) -> dict[str, Any]:
    """No --generator-* flags exposed, so overrides are always empty."""
    return {}


def resolve_backend_version_for_dynamo(
    backend: str,
    backend_version: str | None,
    *args: Any,
    **kwargs: Any,
) -> str | None:
    """Pass-through: keep whatever version the caller already resolved."""
    return backend_version
