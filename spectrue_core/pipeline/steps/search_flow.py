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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class SearchFlowStep:
    """
    Execute search retrieval for target claims.

    Wraps run_search_flow() to collect sources from Tavily/CSE.

    Context Input:
        - extras: target_claims, inline_sources
        - lang, search_type, gpt_model

    Context Output:
        - sources (updated with collected sources)
        - extras: execution_state
    """

    config: Any  # SpectrueConfig
    search_mgr: Any  # SearchManager
    agent: Any  # FactCheckerAgent
    name: str = "search_flow"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute search retrieval."""
        from spectrue_core.verification.pipeline.pipeline_search import (
            SearchFlowInput,
            SearchFlowState,
            run_search_flow,
        )

        try:
            target_claims = ctx.get_extra("target_claims", ctx.claims)
            inline_sources = ctx.get_extra("inline_sources", [])
            search_candidates = ctx.get_extra("search_candidates", []) # Preferred alias
            
            # Prefer search_candidates. If empty, maybe fall back to target_claims?
            # If both empty, skip.
            claims_to_search = search_candidates if search_candidates else target_claims
            
            if not claims_to_search:
                 Trace.event("search_flow.skipped", {"reason": "no_candidates"})
                 return ctx

            progress_callback = ctx.get_extra("progress_callback")


            def can_add_search(model: str, search_type: str, max_cost: int | None) -> bool:
                # Simple budget check
                return True

            inp = SearchFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                max_cost=ctx.get_extra("max_cost"),
                article_intent=ctx.get_extra("article_intent", "general"),
                search_queries=ctx.get_extra("search_queries", []),
                claims=target_claims,
                preloaded_context=ctx.get_extra("prepared_context"),
                progress_callback=progress_callback,
                inline_sources=inline_sources,
                pipeline=ctx.mode.name,
            )

            state = SearchFlowState(
                final_context="",
                final_sources=[],
                preloaded_context=ctx.get_extra("prepared_context"),
                used_orchestration=False,
            )

            result_state = await run_search_flow(
                config=self.config,
                search_mgr=self.search_mgr,
                agent=self.agent,
                can_add_search=can_add_search,
                inp=inp,
                state=state,
            )

            Trace.event(
                "search_flow.step_completed",
                {
                    "sources_collected": len(result_state.final_sources),
                    "used_orchestration": result_state.used_orchestration,
                },
            )

            return (
                ctx.with_update(sources=result_state.final_sources)
                .set_extra("search_context", result_state.final_context)
                .set_extra("execution_state", result_state.execution_state)
            )

        except Exception as e:
            logger.exception("[SearchFlowStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
