# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
- execute_pipeline: Entry point for pipeline execution

Example:
    from spectrue_core.pipeline import execute_pipeline

    # Execute claims through pipeline
    result = await execute_pipeline(
        mode_name="normal",
        claims=claims,
        search_mgr=search_mgr,
        agent=agent,
    )
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
from spectrue_core.pipeline.constants import (
    DAG_EXECUTION_STATE_KEY,
    DAG_EXECUTION_SUMMARY_KEY,
    DAG_STEP_STATUS_PENDING,
    DAG_STEP_STATUS_RUNNING,
    DAG_STEP_STATUS_SUCCEEDED,
    DAG_STEP_STATUS_FAILED,
    DAG_STEP_STATUS_SKIPPED,
)
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.executor import (
    execute_pipeline,
    validate_claims_for_mode,
)


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
    # DAG constants
    "DAG_EXECUTION_STATE_KEY",
    "DAG_EXECUTION_SUMMARY_KEY",
    "DAG_STEP_STATUS_PENDING",
    "DAG_STEP_STATUS_RUNNING",
    "DAG_STEP_STATUS_SUCCEEDED",
    "DAG_STEP_STATUS_FAILED",
    "DAG_STEP_STATUS_SKIPPED",
    # Factory
    "PipelineFactory",
    # Executor
    "execute_pipeline",
    "validate_claims_for_mode",
    # Errors
    "PipelineViolation",
    "PipelineExecutionError",
]
