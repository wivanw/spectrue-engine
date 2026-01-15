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
import uuid
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.pipeline.contracts import SEARCH_PLAN_KEY, SearchPlan
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.coverage_anchors import extract_all_anchors
from spectrue_core.verification.pipeline.pipeline_queries import (
    normalize_and_sanitize,
    resolve_budgeted_max_queries,
    select_diverse_queries,
)
from spectrue_core.verification.retrieval.cegs_mvp import build_doc_query_plan
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)

logger = logging.getLogger(__name__)


def _fallback_fact(ctx: PipelineContext) -> str:
    for key in ("prepared_fact", "clean_text", "input_text", "original_fact"):
        val = ctx.get_extra(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _append_query(queries: list[str], candidate: str | None) -> None:
    if not candidate:
        return
    normalized = normalize_and_sanitize(candidate)
    if not normalized:
        return
    if normalized in queries:
        return
    queries.append(normalized)


@dataclass
class BuildClusterQueriesStep:
    """Build clustered retrieval query plans for deep_v2."""

    name: str = "build_cluster_queries"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            clusters = ctx.get_extra("cluster_representatives", {}) or {}
            cluster_claims = ctx.get_extra("cluster_claims", {}) or {}

            if not clusters:
                Trace.event("retrieval.cluster_plan.skipped", {"reason": "no_clusters"})
                return ctx

            profile_name = resolve_profile_name(ctx.mode.name)
            profile = default_search_policy().get_profile(profile_name)

            full_text = ctx.get_extra("prepared_fact", "") or ctx.get_extra("input_text", "")
            
            # Reuse cached anchors if already extracted by claim extraction step
            cached_anchors = ctx.get_extra("extracted_anchors")
            if cached_anchors is not None:
                anchors = cached_anchors
                Trace.event("claims.coverage.anchors.cached", {"count": len(anchors)})
            else:
                anchors = extract_all_anchors(full_text)
            
            fact_fallback = _fallback_fact(ctx)

            cluster_plans: list[dict[str, Any]] = []
            all_queries: list[str] = []

            for cluster_id, rep_claims in clusters.items():
                rep_claims_list = [c for c in (rep_claims or []) if isinstance(c, dict)]
                claims_for_plan = rep_claims_list or [
                    c for c in (cluster_claims.get(cluster_id, []) or []) if isinstance(c, dict)
                ]
                max_queries = resolve_budgeted_max_queries(claims_for_plan, default_max=3)

                try:
                    cluster_queries = build_doc_query_plan(claims_for_plan, anchors)
                    if not cluster_queries:
                        Trace.event(
                            "retrieval.cluster_plan.empty",
                            {"cluster_id": cluster_id, "reason": "no_queries"},
                        )
                        cluster_queries = select_diverse_queries(
                            claims_for_plan,
                            max_queries=max_queries,
                            fact_fallback=fact_fallback,
                        )
                except Exception as exc:
                    logger.warning("Cluster query planning failed: %s", exc)
                    cluster_queries = select_diverse_queries(
                        claims_for_plan,
                        max_queries=max_queries,
                        fact_fallback=fact_fallback,
                    )

                deduped: list[str] = []
                for query in cluster_queries:
                    _append_query(deduped, query)
                if not deduped:
                    _append_query(deduped, fact_fallback)

                all_queries.extend(deduped)

                cluster_plans.append(
                    {
                        "cluster_id": cluster_id,
                        "representative_claim_ids": [
                            str(c.get("id") or c.get("claim_id"))
                            for c in rep_claims_list
                            if isinstance(c, dict)
                        ],
                        "search_queries": deduped,
                        "trace": {
                            "profile": profile.name,
                            "search_depth": profile.search_depth,
                            "max_results": profile.max_results,
                            "max_queries": max_queries,
                        },
                    }
                )

            plan_id = uuid.uuid4().hex[:10]
            plan = SearchPlan(
                plan_id=plan_id,
                mode=AnalysisMode.DEEP_V2.value,
                global_queries=tuple(all_queries),
                per_claim_queries={},
                trace={
                    "profile": profile.name,
                    "max_results": profile.max_results,
                    "cluster_count": len(cluster_plans),
                },
            )

            Trace.event(
                "retrieval.cluster_plan",
                {
                    "plan_id": plan_id,
                    "clusters": len(cluster_plans),
                    "queries": len(all_queries),
                },
            )

            return (
                ctx.set_extra("cluster_search_plans", cluster_plans)
                .set_extra(SEARCH_PLAN_KEY, plan)
                .set_extra(
                    "retrieval_plan_trace",
                    {
                        "plan_id": plan_id,
                        "clusters": len(cluster_plans),
                        "queries": len(all_queries),
                    },
                )
            )
        except Exception as exc:
            logger.exception("[BuildClusterQueriesStep] Failed: %s", exc)
            raise PipelineExecutionError(self.name, str(exc), cause=exc) from exc
