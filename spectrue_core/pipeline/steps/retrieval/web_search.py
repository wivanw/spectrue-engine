# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.contracts import (
    RAW_SEARCH_RESULTS_KEY,
    SEARCH_PLAN_KEY,
    RawSearchResult,
    RawSearchResults,
    SearchPlan,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def _group_sources_by_claim(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        claim_id = source.get("claim_id")
        if not claim_id:
            continue
        grouped.setdefault(str(claim_id), []).append(source)
    return grouped


@dataclass
class WebSearchStep:
    """Execute web search based on the current query plan."""

    config: Any
    search_mgr: Any
    agent: Any
    name: str = "web_search"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            from spectrue_core.verification.pipeline.pipeline_search import (
                SearchFlowInput,
                SearchFlowState,
                run_search_flow,
            )

            plan: SearchPlan | None = ctx.get_extra(SEARCH_PLAN_KEY)
            plan_id = plan.plan_id if plan else "plan-unknown"
            global_queries = list(plan.global_queries) if plan else ctx.get_extra("search_queries", [])

            target_claims = ctx.get_extra("target_claims", ctx.claims)
            inline_sources = ctx.get_extra("inline_sources", [])
            search_candidates = ctx.get_extra("search_candidates", [])
            claims_to_search = search_candidates if search_candidates else target_claims

            if not claims_to_search:
                Trace.event("retrieval.search.skipped", {"reason": "no_candidates", "plan_id": plan_id})
                empty_results = RawSearchResults(plan_id=plan_id, results=tuple())
                return ctx.set_extra(RAW_SEARCH_RESULTS_KEY, empty_results)

            progress_callback = ctx.get_extra("progress_callback")

            def can_add_search(model: str, search_type: str, max_cost: int | None) -> bool:
                return True

            inp = SearchFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                max_cost=ctx.get_extra("max_cost"),
                article_intent=ctx.get_extra("article_intent", "general"),
                search_queries=global_queries,
                claims=claims_to_search,
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

            sources = list(result_state.final_sources)
            grouped = _group_sources_by_claim(sources)
            ungrouped = [src for src in sources if not isinstance(src, dict) or not src.get("claim_id")]
            raw_results: list[RawSearchResult] = []

            if grouped:
                for claim_id, claim_sources in grouped.items():
                    query = ""
                    if plan and plan.per_claim_queries.get(claim_id):
                        query = plan.per_claim_queries[claim_id][0]
                    elif global_queries:
                        query = global_queries[0]
                    raw_results.append(
                        RawSearchResult(
                            plan_id=plan_id,
                            query_id=f"{claim_id}:legacy",
                            query=query,
                            claim_id=claim_id,
                            provider_payload=claim_sources,
                            trace={
                                "source": "legacy_search_flow",
                                "sources_count": len(claim_sources),
                            },
                        )
                    )
                if ungrouped:
                    query = global_queries[0] if global_queries else ""
                    raw_results.append(
                        RawSearchResult(
                            plan_id=plan_id,
                            query_id="global:legacy",
                            query=query,
                            claim_id=None,
                            provider_payload=ungrouped,
                            trace={
                                "source": "legacy_search_flow",
                                "sources_count": len(ungrouped),
                            },
                        )
                    )
            else:
                query = global_queries[0] if global_queries else ""
                raw_results.append(
                    RawSearchResult(
                        plan_id=plan_id,
                        query_id="global:legacy",
                        query=query,
                        claim_id=None,
                        provider_payload=sources,
                        trace={
                            "source": "legacy_search_flow",
                            "sources_count": len(sources),
                        },
                    )
                )

            raw_contract = RawSearchResults(
                plan_id=plan_id,
                results=tuple(raw_results),
                trace={
                    "used_orchestration": result_state.used_orchestration,
                    "sources_total": len(sources),
                },
            )

            Trace.event(
                "retrieval.search",
                {
                    "plan_id": plan_id,
                    "sources_count": len(sources),
                    "used_orchestration": result_state.used_orchestration,
                },
            )

            return (
                ctx.with_update(sources=sources)
                .set_extra("search_context", result_state.final_context)
                .set_extra("execution_state", result_state.execution_state)
                .set_extra(RAW_SEARCH_RESULTS_KEY, raw_contract)
                .set_extra(
                    "retrieval_search_trace",
                    {
                        "plan_id": plan_id,
                        "sources": len(sources),
                    },
                )
            )

        except Exception as e:
            logger.exception("[WebSearchStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
