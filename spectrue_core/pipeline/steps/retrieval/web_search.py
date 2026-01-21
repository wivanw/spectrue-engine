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
from spectrue_core.verification.orchestration.sufficiency import SUFFICIENCY_P_THRESHOLD
from spectrue_core.verification.retrieval.fixed_pipeline import (
    ExtractedContent,
    FixedPipelineContext,
    bind_after_extract,
    compute_sufficiency,
    current_stage,
    extract_all_batches,
    init_state,
    normalize_url,
    register_urls,
    source_id_for_url,
)
from spectrue_core.verification.retrieval.cegs_mvp import (
    EvidenceItem,
    EvidencePool,
    EvidenceSourceMeta,
    _compute_content_hash,
    match_claim_to_pool,
)

logger = logging.getLogger(__name__)


def _claim_id_for(claim: dict[str, Any], idx: int) -> str:
    raw = claim.get("id") or claim.get("claim_id")
    if raw:
        return str(raw)
    return f"c{idx + 1}"


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return score


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return {}


def _first_query_for_claim(
    plan: SearchPlan | None,
    claim_id: str,
    claim: dict[str, Any],
) -> str | None:
    if plan and plan.per_claim_queries:
        queries = list(plan.per_claim_queries.get(claim_id) or [])
        for q in queries:
            if isinstance(q, str) and q.strip():
                return q
    raw_queries = claim.get("search_queries") or []
    if isinstance(raw_queries, list):
        for q in raw_queries:
            if isinstance(q, str) and q.strip():
                return q
    return None


def _record_sources(
    sources: list[dict[str, Any]],
    url_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    urls: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = source.get("url") or source.get("link")
        if not url:
            continue
        nurl = normalize_url(str(url))
        if not nurl:
            continue
        source_id = source_id_for_url(nurl)
        urls.append(nurl)
        if nurl not in url_metadata:
            score = source.get("score")
            if score is None:
                score = source.get("relevance_score")
            if score is None:
                score = source.get("provider_score")
            url_metadata[nurl] = {
                "url": nurl,
                "title": source.get("title") or "",
                "snippet": source.get("content") or source.get("snippet") or "",
                "score": _coerce_score(score),
                "source_id": source_id,
                "provider_meta": dict(source, source_id=source_id),
            }
    return urls


def _apply_metadata(
    extracted: dict[str, ExtractedContent],
    url_metadata: dict[str, dict[str, Any]],
) -> None:
    for url, content in extracted.items():
        meta = url_metadata.get(url)
        if meta:
            content.metadata.update(meta)
        content.metadata.setdefault("url", url)
        content.metadata.setdefault("source_id", meta.get("source_id") if meta else source_id_for_url(url))


def _build_evidence_pool(
    extracted: dict[str, ExtractedContent],
    url_metadata: dict[str, dict[str, Any]],
) -> EvidencePool:
    pool = EvidencePool()
    items: list[EvidenceItem] = []
    for url, content in extracted.items():
        meta = url_metadata.get(url, {})
        source_meta = EvidenceSourceMeta(
            url=url,
            title=str(meta.get("title") or ""),
            snippet=str(meta.get("snippet") or ""),
            score=_coerce_score(meta.get("score")),
            provider_meta=dict(meta.get("provider_meta") or {}),
        )
        items.append(
            EvidenceItem(
                url=url,
                extracted_text=str(content.text or ""),
                citations=[],
                content_hash=_compute_content_hash(str(content.text or "")),
                source_meta=source_meta,
            )
        )
    if items:
        pool.add_items(items)
    return pool


def _build_sources_for_urls(
    urls: list[str],
    extracted: dict[str, ExtractedContent],
    url_metadata: dict[str, dict[str, Any]],
    *,
    claim_id: str | None = None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for url in urls:
        content = extracted.get(url)
        meta = url_metadata.get(url, {})
        src = dict(meta.get("provider_meta") or {})
        src["url"] = url
        src.setdefault("source_id", meta.get("source_id") or source_id_for_url(url))
        src["title"] = meta.get("title") or ""
        src["snippet"] = meta.get("snippet") or ""
        if content and content.text:
            src["content"] = content.text
            src["fulltext"] = True
        score = meta.get("score")
        if score is not None:
            src["provider_score"] = score
            src["score"] = score
        if claim_id is not None:
            src["claim_id"] = str(claim_id)
        sources.append(src)
    return sources


@dataclass
class WebSearchStep:
    """Execute web search based on the current query plan."""

    config: Any
    search_mgr: Any
    agent: Any
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

            claim_id_map: dict[str, dict[str, Any]] = {}
            for idx, claim in enumerate(safe_claims):
                claim_id_map[_claim_id_for(claim, idx)] = claim

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

            async def _run_search_queries(queries: list[str]) -> list[str]:
                urls: list[str] = []
                for query in queries:
                    if not query:
                        continue
                    _, sources = await self.search_mgr.search_phase(
                        query,
                        max_results=5,
                        depth="basic",
                        topic="general",
                    )
                    urls.extend(_record_sources(sources, url_metadata))
                return urls

            def _refresh_audit_matches() -> None:
                nonlocal audited_urls_by_claim
                pool = _build_evidence_pool(state.extractor_queue.extracted, url_metadata)
                audited_urls_by_claim = {}
                for idx, claim in enumerate(safe_claims):
                    claim_id = _claim_id_for(claim, idx)
                    bundle = match_claim_to_pool(claim, pool)
                    audited_urls_by_claim[claim_id] = {item.url for item in bundle.matched_items}

            with FixedPipelineContext(
                state=state,
                extract_batch=_extract_batch,
                audit_match=_audit_match,
            ) as pipeline_ctx:
                # Stage 0: Universal search
                pipeline_ctx.set_stage(0)
                stage0_urls = await _run_search_queries(list(plan.global_queries))
                register_urls(stage=0, claim_ids=all_claim_ids, urls=stage0_urls)
                await extract_all_batches()
                _apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Stage 1: Graph priority
                pipeline_ctx.set_stage(1)
                key_claim_ids = ctx.get_extra("key_claim_ids", []) or []
                for claim_id in key_claim_ids:
                    claim = claim_id_map.get(str(claim_id))
                    if not claim:
                        continue
                    query = _first_query_for_claim(plan, str(claim_id), claim)
                    if not query:
                        continue
                    stage1_urls = await _run_search_queries([query])
                    register_urls(stage=1, claim_ids={str(claim_id)}, urls=stage1_urls)
                await extract_all_batches()
                _apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Stage 2: Sufficiency-driven escalation
                pipeline_ctx.set_stage(2)
                for idx, claim in enumerate(safe_claims):
                    claim_id = _claim_id_for(claim, idx)
                    metadata = _metadata_dict(claim.get("metadata")) or _metadata_dict(claim)
                    s_value = compute_sufficiency(metadata)
                    s_min = float(metadata.get("S_min", SUFFICIENCY_P_THRESHOLD))
                    if s_value < s_min:
                        query = _first_query_for_claim(plan, claim_id, claim)
                        if not query:
                            continue
                        stage2_urls = await _run_search_queries([query])
                        register_urls(stage=2, claim_ids={claim_id}, urls=stage2_urls)
                await extract_all_batches()
                _apply_metadata(state.extractor_queue.extracted, url_metadata)
                _refresh_audit_matches()
                bind_after_extract()

                # Record final Bayesian sufficiency for each claim
                from spectrue_core.verification.orchestration.sufficiency import check_sufficiency_for_claim
                for claim_id, claim in claim_id_map.items():
                    # bind_after_extract ensures state.bindings.audited is up to date
                    claim_urls = state.bindings.audited.get(claim_id, set())
                    claim_sources = _build_sources_for_urls(
                        list(claim_urls),
                        state.extractor_queue.extracted,
                        url_metadata,
                        claim_id=claim_id
                    )
                    res = check_sufficiency_for_claim(claim, claim_sources)
                    claim_sufficiency[claim_id] = res.posterior_p

            # Build sources for downstream steps
            extracted_urls = list(state.extractor_queue.extracted.keys())
            global_sources = _build_sources_for_urls(
                extracted_urls,
                state.extractor_queue.extracted,
                url_metadata,
            )

            by_claim_sources: dict[str, list[dict[str, Any]]] = {}
            for claim_id, urls in state.bindings.audited.items():
                ordered = [u for u in extracted_urls if u in urls]
                by_claim_sources[str(claim_id)] = _build_sources_for_urls(
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

            audit_sources = _build_sources_for_urls(
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
