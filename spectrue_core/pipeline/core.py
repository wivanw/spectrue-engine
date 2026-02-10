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
Pipeline Core

Defines the Step protocol and Pipeline executor.

Design Principles:
- Steps are composable units of work with a single run() method
- Pipeline executes steps in order, threading context through
- Steps are stateless; state lives in PipelineContext
- Mode-specific behavior is achieved through different step compositions

Usage:
    from spectrue_core.pipeline import Pipeline, Step, PipelineContext

    class MyStep(Step):
        name = "my_step"

        async def run(self, ctx: PipelineContext) -> PipelineContext:
            # do work
            return ctx.with_update(result=my_result)

    pipeline = Pipeline(mode=GENERAL_MODE, steps=[MyStep()])
    result = await pipeline.run(initial_context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from spectrue_core.use_cases.verification.pipeline_types import PipelineMode, PipelineContext, Step
from spectrue_core.pipeline.errors import PipelineExecutionError, PipelineViolation


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Context
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Executor
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Pipeline:
    """
    Executes a sequence of steps, threading context through.

    The pipeline is the composition of steps for a specific mode.
    Different modes have different step sequences, but the executor
    logic is the same.

    Attributes:
        mode: Pipeline mode configuration
        steps: Ordered list of steps to execute

    Example:
        pipeline = Pipeline(
            mode=GENERAL_MODE,
            steps=[
                AssertSingleClaimStep(),
                RetrievalStep(),
                ScoringStep(),
            ]
        )
        result = await pipeline.run(initial_ctx)
    """

    mode: PipelineMode
    steps: list[Step]

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute all steps in order.

        Each step receives the context from the previous step.
        If any step fails, execution stops and the error propagates.

        Args:
            ctx: Initial pipeline context

        Returns:
            Final context after all steps complete

        Raises:
            PipelineViolation: If an invariant check fails
            PipelineExecutionError: If step execution fails
        """
        trace = ctx.trace

        if trace:
            trace.event(
                "pipeline_start",
                mode=self.mode.name,
                step_count=len(self.steps),
                step_names=[s.name for s in self.steps],
            )

        current_ctx = ctx

        for i, step in enumerate(self.steps):
            step_name = step.name

            if trace:
                trace.event("pipeline_step_start", step=step_name, index=i)

            try:
                current_ctx = await step.run(current_ctx)
            except Exception as e:
                if trace:
                    trace.event(
                        "pipeline_step_error",
                        step=step_name,
                        index=i,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                # Re-raise PipelineViolation and PipelineExecutionError as-is
                if isinstance(e, (PipelineViolation, PipelineExecutionError)):
                    raise
                # Wrap other exceptions
                raise PipelineExecutionError(step_name, str(e), cause=e) from e

            if trace:
                trace.event("pipeline_step_end", step=step_name, index=i)

        if trace:
            trace.event("pipeline_end", mode=self.mode.name)

        return current_ctx

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"Pipeline(mode={self.mode.name}, steps={step_names})"
