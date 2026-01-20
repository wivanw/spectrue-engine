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

from spectrue_core.pipeline.contracts import SEARCH_PLAN_KEY, SearchPlan
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.coverage_anchors import extract_all_anchors
from spectrue_core.verification.pipeline.pipeline_queries import (
    is_fuzzy_duplicate,
    normalize_and_sanitize,
    resolve_budgeted_max_queries,
    select_diverse_queries,
)
from spectrue_core.verification.retrieval.cegs_mvp import build_doc_query_plan
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)
from spectrue_core.verification.search.search_escalation import (
    build_query_variants,
    trace_query_variants,
)

logger = logging.getLogger(__name__)


def _claim_id_for(claim: dict[str, Any], idx: int) -> str:
    raw = claim.get("id") or claim.get("claim_id")
    if raw:
        return str(raw)
    return f"c{idx + 1}"


def _fallback_fact(ctx: PipelineContext) -> str:
    for key in ("prepared_fact", "clean_text", "input_text", "original_fact"):
        val = ctx.get_extra(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _append_query(
    queries: list[str],
    candidate: str | None,
) -> None:
    if not candidate:
        return
    normalized = normalize_and_sanitize(candidate)
    if not normalized:
        return
    if is_fuzzy_duplicate(normalized, queries, threshold=0.9):
        return
    queries.append(normalized)


def _build_claim_queries(claim: dict[str, Any], max_queries: int) -> list[str]:
    """
    Build search queries from claim data.
    
    Prioritizes retrieval_seed_terms over search_queries and query_candidates.
    Seed terms are joined into a keyword query (not full sentences).
    """
    queries: list[str] = []

    # Priority 1 - retrieval_seed_terms (joined as keyword query)
    seed_terms = claim.get("retrieval_seed_terms")
    if seed_terms and isinstance(seed_terms, list):
        valid_terms = [t for t in seed_terms if isinstance(t, str) and len(t) >= 2]
        if len(valid_terms) >= 3:
            # Join first 6 seed terms into a keyword query
            keyword_query = " ".join(valid_terms[:6])
            _append_query(queries, keyword_query)
            if len(queries) >= max_queries:
                return queries

    # Priority 2 - search_queries (from enrichment)
    for raw in claim.get("search_queries", []) or []:
        _append_query(queries, raw if isinstance(raw, str) else None)
        if len(queries) >= max_queries:
            return queries

    # Priority 3 - query_candidates
    for candidate in claim.get("query_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        _append_query(queries, candidate.get("text"))
        if len(queries) >= max_queries:
            return queries

    # Priority 4 - fallback to normalized_text/text
    fallback = claim.get("normalized_text") or claim.get("text")
    if fallback:
        _append_query(queries, fallback)

    return queries[:max_queries]


@dataclass
class BuildQueriesStep:
    """Build a structured query plan without executing search."""

    name: str = "build_queries"
    weight: float = 3.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        claims = ctx.claims or []
        claims_for_plan = ctx.get_extra("target_claims", claims) or []
        # Use api_analysis_mode for consistent mode string
        mode = str(ctx.mode.api_analysis_mode)

        
        profile_name = resolve_profile_name(ctx.mode.name)
        policy = default_search_policy()
        profile = policy.get_profile(profile_name)

        max_queries = resolve_budgeted_max_queries(claims_for_plan, default_max=3)
        fact_fallback = _fallback_fact(ctx)

        # CEGS MVP Integration: Use doc-level query planning (A)
        # Extract anchors from full text if available (with caching)
        full_text = ctx.get_extra("prepared_fact", "") or ctx.get_extra("input_text", "")
        
        # Reuse cached anchors if already extracted by claim extraction step
        cached_anchors = ctx.get_extra("extracted_anchors")
        if cached_anchors is not None:
            anchors = cached_anchors
            Trace.event("claims.coverage.anchors.cached", {"count": len(anchors)})
        else:
            anchors = extract_all_anchors(full_text)
            # Note: anchor trace event is emitted inside extract_all_anchors
        
        # Ensure claims are dicts
        safe_claims = [c for c in claims_for_plan if isinstance(c, dict)]
        
        try:
            # Use CEGS planner
            cegs_queries = build_doc_query_plan(safe_claims, anchors)
            
            # (A) Handle empty doc-plan with fallback
            if not cegs_queries:
                Trace.event("retrieval.doc_plan.empty", {
                    "reason": "no_queries",
                    "entity_count_used": 0,
                    "anchor_count_used": len(anchors),
                })
                
                # Fallback to legacy query builder
                global_queries = select_diverse_queries(
                    claims_for_plan,
                    max_queries=max_queries,
                    fact_fallback=fact_fallback,
                )
                
                Trace.event("retrieval.doc_plan.fallback", {
                    "fallback_used": True,
                    "queries_count": len(global_queries),
                    "method": "legacy",
                })
            else:
                global_queries = cegs_queries
                
        except Exception as e:
            # Exception fallback to legacy
            logger.warning(f"CEGS query planning failed: {e}. Falling back to legacy.")
            Trace.event("retrieval.doc_plan.empty", {
                "reason": "exception",
                "error": str(e)[:100],
                "anchor_count_used": len(anchors),
            })
            
            global_queries = select_diverse_queries(
                claims_for_plan,
                max_queries=max_queries,
                fact_fallback=fact_fallback,
            )
            
            Trace.event("retrieval.doc_plan.fallback", {
                "fallback_used": True,
                "queries_count": len(global_queries),
                "method": "legacy_exception",
            })
        
        # (A) Terminal guard: assert global_queries_count >= 1
        if not global_queries:
            Trace.event("retrieval.doc_plan.terminal_empty", {
                "claims_count": len(claims_for_plan),
                "anchors_count": len(anchors),
            })
            raise ValueError("PIPELINE_ERROR: Retrieval planning produced 0 global queries")

        per_claim_queries: dict[str, tuple[str, ...]] = {}
        for idx, claim in enumerate(claims_for_plan):
            if not isinstance(claim, dict):
                continue
            claim_id = _claim_id_for(claim, idx)
            queries = _build_claim_queries(claim, max_queries=max_queries)
            per_claim_queries[claim_id] = tuple(queries)
        
        # ... (rest of the function)
        updated_claims: list[dict[str, Any]] = []
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            claim_id = _claim_id_for(claim, idx)
            queries = per_claim_queries.get(claim_id)
            claim_copy = dict(claim)
            if queries:
                claim_copy["search_queries"] = list(queries)
            
            # Add query variants for escalation ladder
            variants = build_query_variants(claim)
            if variants:
                claim_copy["query_variants"] = [v.to_dict() for v in variants]
                trace_query_variants(claim_id, variants)
            
            updated_claims.append(claim_copy)

        plan_id = uuid.uuid4().hex[:10]
        plan = SearchPlan(
            plan_id=plan_id,
            mode=mode,
            global_queries=tuple(global_queries),
            per_claim_queries=per_claim_queries,
            trace={
                "profile": profile.name,
                "max_results": profile.max_results,
                "max_queries": max_queries,
                "claims": len(claims_for_plan),
            },
        )

        Trace.event(
            "retrieval.plan",
            {
                "plan_id": plan_id,
                "mode": mode,
                "global_queries": len(global_queries),
                "claims": len(claims_for_plan),
                "per_claim_queries": len(per_claim_queries),
            },
        )

        next_ctx = ctx
        if updated_claims:
            next_ctx = next_ctx.with_update(claims=updated_claims)
            target_claims = ctx.get_extra("target_claims")
            if isinstance(target_claims, list):
                updated_targets: list[dict[str, Any]] = []
                for idx, claim in enumerate(target_claims):
                    if not isinstance(claim, dict):
                        continue
                    claim_id = _claim_id_for(claim, idx)
                    queries = per_claim_queries.get(claim_id)
                    claim_copy = dict(claim)
                    if queries:
                        claim_copy["search_queries"] = list(queries)
                    updated_targets.append(claim_copy)
                if updated_targets:
                    next_ctx = next_ctx.set_extra("target_claims", updated_targets)

        return (
            next_ctx.set_extra(SEARCH_PLAN_KEY, plan)
            .set_extra("search_queries", list(global_queries))
            .set_extra(
                "retrieval_plan_trace",
                {
                    "plan_id": plan_id,
                    "global_queries": len(global_queries),
                    "claims": len(claims_for_plan),
                },
            )
        )
