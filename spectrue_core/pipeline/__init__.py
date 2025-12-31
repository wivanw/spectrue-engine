# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Pipeline Module

Step-based pipeline composition for verification workflows.

This module provides:
- PipelineMode: Single source of truth for mode invariants
- Step: Protocol for composable pipeline steps
- Pipeline: Executor that runs steps in sequence
- PipelineContext: Immutable context threaded through steps
- PipelineFactory: Builds pipelines with correct step composition

Example:
    from spectrue_core.pipeline import (
        PipelineFactory,
        PipelineContext,
        NORMAL_MODE,
    )

    # Build a pipeline for normal mode
    factory = PipelineFactory(search_mgr=search_mgr, agent=agent)
    pipeline = factory.build("normal")

    # Execute
    ctx = PipelineContext(mode=NORMAL_MODE, claims=claims)
    result = await pipeline.run(ctx)
"""

from spectrue_core.pipeline.mode import (
    PipelineMode,
    NORMAL_MODE,
    DEEP_MODE,
    get_mode,
)
from spectrue_core.pipeline.core import (
    PipelineContext,
    Step,
    Pipeline,
)
from spectrue_core.pipeline.errors import (
    PipelineViolation,
    PipelineExecutionError,
)
from spectrue_core.pipeline.factory import PipelineFactory


__all__ = [
    # Mode
    "PipelineMode",
    "NORMAL_MODE",
    "DEEP_MODE",
    "get_mode",
    # Core
    "PipelineContext",
    "Step",
    "Pipeline",
    # Factory
    "PipelineFactory",
    # Errors
    "PipelineViolation",
    "PipelineExecutionError",
]
