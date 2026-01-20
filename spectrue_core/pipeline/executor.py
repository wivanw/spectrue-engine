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
Pipeline Executor

Entry point for executing claims through the Step-based pipeline.
This module bridges the legacy flow with the new Pipeline architecture.

Usage:
    # Opt-in: use via feature flag or explicit call
    from spectrue_core.pipeline.executor import execute_pipeline

    result = await execute_pipeline(
        mode="general",
        claims=claims,
        search_mgr=search_mgr,
        agent=agent,
        lang="en",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.pipeline import (
    PipelineContext,
    PipelineFactory,
    PipelineViolation,
    get_mode,
)
from spectrue_core.pipeline.constants import (
    DAG_EXECUTION_STATE_KEY,
    DAG_EXECUTION_SUMMARY_KEY,
)
from spectrue_core.utils.trace import Trace


logger = logging.getLogger(__name__)


async def execute_pipeline(
    *,
    mode_name: str,
    claims: list[dict],
    search_mgr: Any,
    agent: Any,
    lang: str = "en",
    max_cost: int | None = None,
    inline_sources: list[dict] | None = None,
    progress_callback: Any | None = None,
    trace: Any | None = None,
) -> dict[str, Any]:
    """
    Execute claims through the Step-based pipeline.

    This is the new entry point that replaces direct PhaseRunner calls.
    It builds the correct pipeline for the mode and executes all steps.

    Args:
        mode_name: "general", "deep", or "deep_v2"
        claims: List of claim dicts to process
        search_mgr: SearchManager for retrieval
        agent: FactCheckerAgent for LLM operations
        lang: Primary language code
        max_cost: Max cost budget
        inline_sources: Pre-verified inline sources
        progress_callback: Async progress callback
        trace: Trace instance

    Returns:
        Dict with sources, evidence, verdict, etc.

    Raises:
        PipelineViolation: If mode invariants are violated
        PipelineExecutionError: If execution fails
    """
    mode = get_mode(mode_name)

    # Build initial context
    ctx = PipelineContext(
        mode=mode,
        claims=claims,
        lang=lang,
        trace=trace,
        progress_callback=progress_callback,
    )

    # Set extras for steps that need them
    ctx = ctx.set_extra("max_cost", max_cost)
    ctx = ctx.set_extra("inline_sources", inline_sources or [])

    # Build pipeline for this mode
    factory = PipelineFactory(search_mgr=search_mgr, agent=agent)
    pipeline = factory.build(mode_name)

    Trace.event(
        "pipeline_executor.start",
        {
            "mode": mode.name,
            "claims_count": len(claims),
            "step_count": len(pipeline.steps),
            "step_names": [s.name for s in pipeline.steps],
        },
    )

    try:
        # Execute pipeline
        result_ctx = await pipeline.run(ctx)

        Trace.event(
            "pipeline_executor.completed",
            {
                "mode": mode.name,
                "sources_count": len(result_ctx.sources),
                "has_verdict": result_ctx.verdict is not None,
            },
        )

        # Convert context to result dict
        final_result = result_ctx.get_extra("final_result")
        if isinstance(final_result, dict):
            result_payload = dict(final_result)
            _attach_execution_metadata(result_payload, result_ctx)
            return result_payload
        return _context_to_result(result_ctx)

    except PipelineViolation as e:
        logger.warning("[PipelineExecutor] Invariant violation: %s", e)
        Trace.event("pipeline_executor.violation", e.to_trace_dict())
        raise

    except Exception as e:
        logger.exception("[PipelineExecutor] Execution failed: %s", e)
        Trace.event(
            "pipeline_executor.error",
            {"error": str(e), "error_type": type(e).__name__},
        )
        raise


def _context_to_result(ctx: PipelineContext) -> dict[str, Any]:
    """Convert PipelineContext to result dict for API response."""
    verdict = ctx.verdict or {}

    result = {
        "sources": ctx.sources,
        "evidence": ctx.evidence,
        "verified_score": verdict.get("verified_score"),
        "danger_score": verdict.get("danger_score"),
        "context_score": verdict.get("context_score"),
        "style_score": verdict.get("style_score"),
        "confidence_score": verdict.get("confidence_score"),
        "explainability_score": verdict.get("explainability_score"),
        "rationale": verdict.get("rationale", ""),
        "claim_verdicts": verdict.get("claim_verdicts", []),
        "status": verdict.get("status", "ok"),
    }
    _attach_execution_metadata(result, ctx)
    return result


def _attach_execution_metadata(result: dict[str, Any], ctx: PipelineContext) -> None:
    """Attach execution metadata without overwriting existing payload keys."""
    execution_state = ctx.get_extra("execution_state")
    if execution_state is not None and "execution_state" not in result:
        result["execution_state"] = execution_state

    evidence_map = ctx.get_extra("evidence_map")
    if evidence_map is not None and "evidence_map" not in result:
        result["evidence_map"] = evidence_map

    dag_summary = ctx.get_extra(DAG_EXECUTION_SUMMARY_KEY)
    if dag_summary is not None and "dag_execution_summary" not in result:
        result["dag_execution_summary"] = dag_summary

    dag_state = ctx.get_extra(DAG_EXECUTION_STATE_KEY)
    if dag_state is not None and "dag_execution_state" not in result:
        result["dag_execution_state"] = dag_state


async def validate_claims_for_mode(
    mode_name: str,
    claims: list[dict],
) -> None:
    """
    Validate claims against mode invariants without full execution.

    Use this for pre-flight validation before expensive operations.

    Args:
        mode_name: "general", "deep", or "deep_v2"
        claims: Claims to validate

    Raises:
        PipelineViolation: If validation fails
    """
    from spectrue_core.pipeline.steps import get_invariant_steps_for_mode

    mode = get_mode(mode_name)
    ctx = PipelineContext(mode=mode, claims=claims)

    invariant_steps = get_invariant_steps_for_mode(mode_name)

    for step in invariant_steps:
        ctx = await step.run(ctx)

    logger.debug(
        "[PipelineExecutor] Validated %d claims for mode '%s'",
        len(claims),
        mode_name,
    )
