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

import asyncio
import logging
from dataclasses import dataclass, field
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
from spectrue_core.pipeline.retrieval.cegs_mvp import match_claim_to_pool
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors
from spectrue_core.pipeline.retrieval.web_search_support import (
    apply_metadata,
    build_evidence_pool,
    build_sources_for_urls,
    claim_id_for,
    first_query_for_claim,
    query_dedup_key,
    record_sources,
)
from spectrue_core.use_cases.claims.sufficiency import (
    SufficiencyStatus,
    check_sufficiency_for_claim,
)
from spectrue_core.use_cases.retrieval.decision_gates import build_extract_filter, sufficiency_sensor_log_odds
from spectrue_core.use_cases.verification.pipeline_queries import is_fuzzy_duplicate
from spectrue_core.utils.retrieval_urls import (
    ExtractedContent,
    FixedPipelineContext,
    bind_after_extract,
    current_stage,
    extract_all_batches,
    init_state,
    register_urls,
)
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class WebSearchStep:
    """Execute web search based on the current query plan."""

    config: Any
    search_mgr: Any
    agent: Any
    decisions: DecisionsConfig = field(default_factory=DecisionsConfig)  # M145 gates; all off by default
    sensors: dict[str, DecisionSensorCalibration] = field(default_factory=uncalibrated_sensors)
    name: str = "web_search"
    weight: float = 25.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            plan: SearchPlan | None = ctx.get_extra(SEARCH_PLAN_KEY)
            if not plan or not plan.global_queries:
                raise PipelineExecutionError(self.name, "Missing search plan")

            claims = ctx.get_extra("target_claims", ctx.claims) or []
            safe_claims = [c for c in claims if isinstance(c, dict)]
            if not safe_claims:
                Trace.event("retrieval.search.skipped", {"reason": "no_claims", "plan_id": plan.plan_id})
                return ctx

            # Read concurrency limit from runtime_config
            runtime_config = ctx.get_extra("runtime_config")
            max_search_concurrency = 4  # default
            if runtime_config and hasattr(runtime_config, "llm"):
                max_search_concurrency = getattr(runtime_config.llm, "max_doc_concurrency", 4)
            search_sem = asyncio.Semaphore(max_search_concurrency)

            claim_id_map: dict[str, dict[str, Any]] = {}
            for idx, claim in enumerate(safe_claims):
                claim_id_map[claim_id_for(claim, idx)] = claim

            all_claim_ids = set(claim_id_map.keys())
            url_metadata: dict[str, dict[str, Any]] = {}
            state = init_state()
            audited_urls_by_claim: dict[str, set[str]] = {}
            claim_sufficiency: dict[str, float] = {}

            async def _extract_batch(urls: list[str]) -> dict[str, ExtractedContent]:
                stage = current_stage()
                content_map = await self.search_mgr.fetch_urls_content_batch(urls, stage=stage)
                extracted: dict[str, ExtractedContent] = {}
                for url, text in content_map.items():
                    if not text:
                        continue
                    extracted[url] = ExtractedContent(text=str(text), metadata={"url": url})
                return extracted

            def _audit_match(claim_id: str, content: ExtractedContent) -> bool:
                url = content.metadata.get("url")
                if not url:
                    return False
                return url in audited_urls_by_claim.get(str(claim_id), set())

            # Every query issued in this step (any stage, global or per-claim),
            # keyed by normalized text. A repeat (exact or fuzzy) awaits the
            # first search instead of paying for the same provider call again.
            query_tasks: dict[str, asyncio.Task[list[str]]] = {}

            async def _search_one(query: str) -> list[str]:
                async with search_sem:
                    _, sources = await self.search_mgr.search_phase(
                        query,
                        max_results=5,
                        depth="basic",
                        topic="general",
                    )
                    return record_sources(sources, url_metadata)

            def _task_for_query(query: str, claim_id: str | None) -> asyncio.Task[list[str]]:
                key = query_dedup_key(query)
                prior_key = key if key in query_tasks else next(
                    (k for k in query_tasks if is_fuzzy_duplicate(key, [k], threshold=0.9)),
                    None,
                )
                if prior_key is not None:
                    Trace.event(
                        "retrieval.search.duplicate_query_skipped",
                        {
                            "plan_id": plan.plan_id,
                            "stage": current_stage(),
                            "claim_id": claim_id,
                            "query": query[:120],
                            "matched_query": prior_key[:120],
                        },
                    )
                    return query_tasks[prior_key]
                task = asyncio.ensure_future(_search_one(query))
                query_tasks[key] = task
                return task

            async def _run_search_queries(
                queries: list[str],
                *,
                claim_id: str | None = None,
            ) -> list[str]:
                valid_queries = [q for q in queries if q and q.strip()]
                if not valid_queries:
                    return []

                tasks: list[asyncio.Task[list[str]]] = []
                for q in valid_queries:
                    task = _task_for_query(q, claim_id)
                    if task not in tasks:
                        tasks.append(task)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                urls: list[str] = []
                for r in results:
                    if isinstance(r, list):
                        urls.extend(r)
                    elif isinstance(r, Exception):
                        logger.warning("[WebSearchStep] Query failed: %s", r)
                return urls

            def _claim_sources(claim_id: str) -> list[dict[str, Any]]:
                # bind_after_extract keeps state.bindings.audited up to date
                claim_urls = state.bindings.audited.get(claim_id, set())
                return build_sources_for_urls(
                    list(claim_urls),
                    state.extractor_queue.extracted,
                    url_metadata,
                    claim_id=claim_id,
                )

            def _refresh_audit_matches() -> None:
                nonlocal audited_urls_by_claim
                pool = build_evidence_pool(state.extractor_queue.extracted, url_metadata)
                audited_urls_by_claim = {}
                for idx, claim in enumerate(safe_claims):
                    claim_id = claim_id_for(claim, idx)
                    bundle = match_claim_to_pool(claim, pool)
                    audited_urls_by_claim[claim_id] = {item.url for item in bundle.matched_items}

            decisions = self.decisions
            content_lang = ctx.lang or "en"
            extract_filter = build_extract_filter(  # G3 (M145)
                self.agent,
                decisions,
                self.sensors,
                url_metadata=url_metadata,
                seen_by_claims=lambda u: state.registry.urls[u].seen_by_claims,
                claim_id_map=claim_id_map,
                lang=content_lang,
            )

            with FixedPipelineContext(
                state=state,
                extract_batch=_extract_batch,
                audit_match=_audit_match,
                extract_filter=extract_filter,
            ) as pipeline_ctx:
                # Stage 0: Universal search
                pipeline_ctx.set_stage(0)
                stage0_urls = await _run_search_queries(list(plan.global_queries))
                register_urls(stage=0, claim_ids=all_claim_ids, urls=stage0_urls)
                await extract_all_batches()
                apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Stage 1: Graph priority (parallel per key claim)
                pipeline_ctx.set_stage(1)
                key_claim_ids = ctx.get_extra("key_claim_ids", []) or []

                async def _stage1_one(cid: str) -> tuple[str, list[str]]:
                    claim = claim_id_map.get(str(cid))
                    if not claim:
                        return str(cid), []
                    query = first_query_for_claim(plan, str(cid), claim)
                    if not query:
                        return str(cid), []
                    urls = await _run_search_queries([query], claim_id=str(cid))
                    return str(cid), urls

                stage1_results = await asyncio.gather(
                    *[_stage1_one(cid) for cid in key_claim_ids],
                    return_exceptions=True,
                )
                for r in stage1_results:
                    if isinstance(r, Exception):
                        logger.warning("[WebSearchStep] Stage 1 query failed: %s", r)
                        continue
                    cid, urls = r
                    if urls:
                        register_urls(stage=1, claim_ids={cid}, urls=urls)
                await extract_all_batches()
                apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Stage 2: Sufficiency-driven escalation (parallel)
                pipeline_ctx.set_stage(2)

                async def _stage2_one(cid: str, claim: dict) -> tuple[str, list[str]]:
                    # Skip escalation for sidefact claims — their lower
                    # sufficiency threshold (0.70) means they reach
                    # SUFFICIENT status faster, so escalation is not worth it.
                    ct = str(claim.get("type") or claim.get("claim_type") or "core")
                    if ct == "sidefact":
                        return cid, []

                    # Escalate only when the evidence already bound to this
                    # claim (stages 0-1) is not sufficient.
                    claim_sources = _claim_sources(cid)
                    res = check_sufficiency_for_claim(claim, claim_sources)
                    if res.status == SufficiencyStatus.INSUFFICIENT:  # G4 (M145): sensor term in the posterior
                        delta = await sufficiency_sensor_log_odds(
                            self.agent, decisions, self.sensors, claim, claim_sources, lang=content_lang, claim_id=cid
                        )
                        if delta:
                            res = check_sufficiency_for_claim(claim, claim_sources, sensor_log_odds=delta)
                    if res.status != SufficiencyStatus.INSUFFICIENT:
                        Trace.event(
                            "retrieval.stage2.skipped",
                            {
                                "plan_id": plan.plan_id,
                                "claim_id": cid,
                                "status": res.status.value,
                                "posterior_p": res.posterior_p,
                            },
                        )
                        return cid, []
                    query = first_query_for_claim(plan, cid, claim)
                    if not query:
                        return cid, []
                    urls = await _run_search_queries([query], claim_id=cid)
                    return cid, urls

                stage2_results = await asyncio.gather(
                    *[
                        _stage2_one(claim_id_for(c, i), c)
                        for i, c in enumerate(safe_claims)
                    ],
                    return_exceptions=True,
                )
                for r in stage2_results:
                    if isinstance(r, Exception):
                        logger.warning("[WebSearchStep] Stage 2 query failed: %s", r)
                        continue
                    cid, urls = r
                    if urls:
                        register_urls(stage=2, claim_ids={cid}, urls=urls)
                await extract_all_batches()
                apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Record final Bayesian sufficiency for each claim
                for claim_id, claim in claim_id_map.items():
                    res = check_sufficiency_for_claim(claim, _claim_sources(claim_id))
                    claim_sufficiency[claim_id] = res.posterior_p

            # Build sources for downstream steps
            extracted_urls = list(state.extractor_queue.extracted.keys())
            global_sources = build_sources_for_urls(
                extracted_urls,
                state.extractor_queue.extracted,
                url_metadata,
            )

            by_claim_sources: dict[str, list[dict[str, Any]]] = {}
            for claim_id, urls in state.bindings.audited.items():
                ordered = [u for u in extracted_urls if u in urls]
                by_claim_sources[str(claim_id)] = build_sources_for_urls(
                    ordered,
                    state.extractor_queue.extracted,
                    url_metadata,
                    claim_id=str(claim_id),
                )

            raw_results: list[RawSearchResult] = []
            if global_sources:
                raw_results.append(
                    RawSearchResult(
                        plan_id=plan.plan_id,
                        query_id="fixed:global",
                        query="universal",
                        claim_id=None,
                        provider_payload=global_sources,
                        trace={"fixed_pipeline": True},
                    )
                )
            for claim_id, sources in by_claim_sources.items():
                if not sources:
                    continue
                raw_results.append(
                    RawSearchResult(
                        plan_id=plan.plan_id,
                        query_id=f"fixed:{claim_id}",
                        query="claim",
                        claim_id=str(claim_id),
                        provider_payload=sources,
                        trace={"fixed_pipeline": True},
                    )
                )

            audit_sources = build_sources_for_urls(
                extracted_urls,
                state.extractor_queue.extracted,
                url_metadata,
            )

            audit_trace_context = {
                "plan_id": plan.plan_id,
                "fixed_pipeline": True,
                "claims_total": len(safe_claims),
            }

            all_sources = list(global_sources)
            for sources in by_claim_sources.values():
                all_sources.extend(sources)

            return (
                ctx.with_update(sources=all_sources)
                .set_extra(
                    RAW_SEARCH_RESULTS_KEY,
                    RawSearchResults(
                        plan_id=plan.plan_id,
                        results=tuple(raw_results),
                        trace={"fixed_pipeline": True},
                    ),
                )
                .set_extra(
                    "retrieval_search_trace",
                    {
                        "plan_id": plan.plan_id,
                        "sources": len(all_sources),
                        "fixed_pipeline": True,
                    },
                )
                .set_extra("audit_sources", audit_sources)
                .set_extra("audit_trace_context", audit_trace_context)
                .set_extra("cluster_sufficiency", claim_sufficiency)  # Reuse key for compatibility
            )

        except Exception as e:
            logger.exception("[WebSearchStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
