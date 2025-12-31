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
        mode="normal",
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
    Pipeline,
    PipelineContext,
    PipelineFactory,
    PipelineViolation,
    get_mode,
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
    search_type: str = "general",
    gpt_model: str = "gpt-5-nano",
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
        mode_name: "normal", "general", or "deep"
        claims: List of claim dicts to process
        search_mgr: SearchManager for retrieval
        agent: FactCheckerAgent for LLM operations
        lang: Primary language code
        search_type: Search type string (legacy compat)
        gpt_model: Model to use
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
        search_type=search_type,
        gpt_model=gpt_model,
        trace=trace,
    )

    # Set extras for steps that need them
    ctx = ctx.set_extra("max_cost", max_cost)
    ctx = ctx.set_extra("inline_sources", inline_sources or [])
    ctx = ctx.set_extra("progress_callback", progress_callback)

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

    return {
        "sources": ctx.sources,
        "evidence": ctx.evidence,
        "verified_score": verdict.get("verified_score", 0.5),
        "danger_score": verdict.get("danger_score", 0.0),
        "context_score": verdict.get("context_score", 0.5),
        "style_score": verdict.get("style_score", 0.5),
        "confidence_score": verdict.get("confidence_score", 0.0),
        "explainability_score": verdict.get("explainability_score", 0.5),
        "rationale": verdict.get("rationale", ""),
        "claim_verdicts": verdict.get("claim_verdicts", []),
        "execution_state": ctx.get_extra("execution_state"),
        "evidence_map": ctx.get_extra("evidence_map"),
    }


async def validate_claims_for_mode(
    mode_name: str,
    claims: list[dict],
) -> None:
    """
    Validate claims against mode invariants without full execution.

    Use this for pre-flight validation before expensive operations.

    Args:
        mode_name: "normal" or "deep"
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
