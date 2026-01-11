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
    RETRIEVAL_ITEMS_KEY,
    RawSearchResult,
    RawSearchResults,
    SearchPlan,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.cegs_mvp import (
    doc_retrieve_to_pool,
    match_claim_to_pool,
    compute_deficit,
    escalate_claim,
    hydrate_pool_with_content,
    _extract_entities_from_claim,
    _normalize_text,
    EvidenceBundle,
)
from spectrue_core.verification.retrieval.experiment_mode import is_experiment_mode

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


def _claim_id_for(claim: dict[str, Any], idx: int) -> str:
    raw = claim.get("id") or claim.get("claim_id")
    if raw:
        return str(raw)
    return f"c{idx + 1}"

@dataclass
class WebSearchStep:
    """Execute web search based on the current query plan."""

    config: Any
    search_mgr: Any
    agent: Any
    name: str = "web_search"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            # CEGS MVP Integration
            # We check if we have a plan with global queries (doc queries)
            plan: SearchPlan | None = ctx.get_extra(SEARCH_PLAN_KEY)
            
            # Use CEGS flow if we have queries and it looks like a CEGS plan (or just default)
            use_cegs = True # Force for this feature
            
            if use_cegs and plan and plan.global_queries:
                def _evidence_item_to_source_dict(item, *, claim_id: str | None = None) -> dict[str, Any]:
                    """
                    Convert CEGS EvidenceItem into legacy 'source' dict.
                    IMPORTANT: downstream grouping requires claim_id for by-claim packs.
                    """
                    src = dict(item.source_meta.provider_meta or {})
                    src["url"] = item.url
                    src["title"] = item.source_meta.title
                    # keep snippet/content for matchers and trace
                    src["snippet"] = item.source_meta.snippet
                    src["content"] = item.extracted_text
                    src["fulltext"] = True
                    # normalize score field names for downstream consumers
                    src["provider_score"] = item.source_meta.score
                    src["score"] = item.source_meta.score
                    if claim_id is not None:
                        src["claim_id"] = str(claim_id)
                    return src

                doc_queries = list(plan.global_queries)
                claims = ctx.get_extra("target_claims", ctx.claims) or []
                safe_claims = [c for c in claims if isinstance(c, dict)]
                
                # 1. Build Sanity Terms
                all_text_parts = []
                for c in safe_claims:
                    ents = _extract_entities_from_claim(c)
                    all_text_parts.extend(ents)
                    seeds = c.get("retrieval_seed_terms", [])
                    if isinstance(seeds, list):
                        all_text_parts.extend([str(s) for s in seeds])
                        
                sanity_terms = _normalize_text(" ".join(all_text_parts))
                
                # === PASS 1: Search & Collect URLs (Skip Extraction) ===
                
                # 2. Doc Retrieve
                pool, doc_urls = await doc_retrieve_to_pool(
                    doc_queries, 
                    sanity_terms, 
                    self.search_mgr,
                    skip_extraction=True
                )
                
                # 3. Match & Escalate per claim
                final_bundles: dict[str, EvidenceBundle] = {}
                all_urls_to_batch = list(doc_urls)
                
                for idx, claim in enumerate(safe_claims):
                    claim_id = _claim_id_for(claim, idx)
                    
                    bundle = match_claim_to_pool(claim, pool)
                    deficit = compute_deficit(claim, bundle)
                    
                    if deficit.is_deficit or is_experiment_mode():
                        pool, bundle, esc_urls = await escalate_claim(
                            claim, 
                            pool, 
                            self.search_mgr,
                            skip_extraction=True
                        )
                        all_urls_to_batch.extend(esc_urls)
                        
                    final_bundles[claim_id] = bundle
                    
                # === PASS 2: Batch Extract ALL URLs ===
                
                if all_urls_to_batch and hasattr(self.search_mgr, "fetch_urls_content_batch"):
                    Trace.event("retrieval.global_batch.start", {
                         "total_urls": len(all_urls_to_batch),
                         "unique_urls": len(set(all_urls_to_batch))
                    })
                    try:
                        content_map = await self.search_mgr.fetch_urls_content_batch(all_urls_to_batch)
                        
                        # === PASS 3: Hydrate Pool ===
                        hydrate_pool_with_content(pool, content_map)
                        
                    except Exception as e:
                        logger.warning(f"[WebSearchStep] Global batch extract failed: {e}")
                        Trace.event("retrieval.global_batch.error", {"error": str(e)})
                    
                # 4) Build retrieval_items contract so EvidenceCollectStep can produce evidence_by_claim
                #    - global: representative pool sources (optional; deep mode may ignore)
                #    - by_claim: per-claim matched evidence items (MUST include claim_id)
                retrieval_items: dict[str, Any] = {"global": [], "by_claim": {}}

                # Global pack = all extracted pool items (no claim_id)
                global_sources: list[dict[str, Any]] = []
                for item in pool.items:
                    global_sources.append(_evidence_item_to_source_dict(item))
                retrieval_items["global"] = global_sources

                # By-claim packs = matched items per claim (with claim_id)
                by_claim: dict[str, list[dict[str, Any]]] = {}
                for cid, bundle in final_bundles.items():
                    items = []
                    for ev in getattr(bundle, "matched_items", []) or []:
                        items.append(_evidence_item_to_source_dict(ev, claim_id=cid))
                    if items:
                        by_claim[str(cid)] = items
                retrieval_items["by_claim"] = by_claim

                # 5) Provide audit_sources + audit_trace_context for RGBA audit aggregation.
                #    audit_sources are used for clustering/trace in the RGBA audit engine.
                audit_sources: list[dict[str, Any]] = []
                for meta in pool.meta:
                    sm = dict(meta.provider_meta or {})
                    sm["url"] = meta.url
                    sm["title"] = meta.title
                    sm["snippet"] = meta.snippet
                    sm["provider_score"] = meta.score
                    sm["score"] = meta.score
                    audit_sources.append(sm)

                audit_trace_context = {
                    "cegs_mvp": True,
                    "plan_id": plan.plan_id,
                    "doc_queries": doc_queries,
                    "pool_items": len(pool.items),
                    "claims_total": len(safe_claims),
                    "claims_with_bundles": len([1 for b in final_bundles.values() if (b.matched_items or [])]),
                }
                
                # Store bundles in extra for next steps that know CEGS
                ctx = ctx.set_extra("cegs_evidence_bundles", final_bundles)
                
                # Populate legacy sources + required trace markers
                return (
                    ctx.with_update(sources=global_sources)
                    .set_extra(RETRIEVAL_ITEMS_KEY, retrieval_items)
                    .set_extra("audit_sources", audit_sources)
                    .set_extra("audit_trace_context", audit_trace_context)
                    .set_extra(
                        RAW_SEARCH_RESULTS_KEY, 
                        RawSearchResults(
                            plan_id=plan.plan_id,
                            results=tuple(),
                            trace={"cegs_mvp": True, "pool_size": len(pool.items), "by_claim": len(by_claim)}
                        )
                    )
                    .set_extra(
                        "retrieval_search_trace",
                        {
                            "plan_id": plan.plan_id,
                            "sources": len(global_sources),
                            "cegs_mvp": True,
                        },
                    )
                    .set_extra(
                        "retrieval_rerank_trace",
                        {
                            "plan_id": plan.plan_id,
                            "reranked": len(global_sources),
                            "cegs_mvp": True,
                        },
                    )
                )

            # Legacy Fallback
            from spectrue_core.verification.pipeline.pipeline_search import (
                SearchFlowInput,
# ... (rest of legacy code)
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
