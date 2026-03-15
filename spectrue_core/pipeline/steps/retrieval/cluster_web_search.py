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
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.contracts import SEARCH_PLAN_KEY
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.tools.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.utils.trace import Trace
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.use_cases.claims.sufficiency import check_sufficiency_for_claim, SufficiencyStatus
from spectrue_core.utils.retrieval_urls import normalize_url, source_id_for_url
from spectrue_core.domain.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)
from spectrue_core.use_cases.retrieval.clustering import assign_similarity_clusters
from spectrue_core.use_cases.verification.orchestration.execution_state import (
    ClaimExecutionState,
    RetrievalHop,
)

logger = logging.getLogger(__name__)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return score


def _coerce_int(value: Any, *, default: int, min_v: int = 0, max_v: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default
    if out < min_v:
        out = min_v
    if max_v is not None and out > max_v:
        out = max_v
    return out


def _stable_cluster_id(urls: list[str]) -> str:
    ordered = sorted([u for u in urls if u])
    raw = "|".join(ordered)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


@dataclass
class ClusterWebSearchStep:
    """Execute cluster-level search/extract for deep_v2."""

    config: Any
    search_mgr: Any
    embedding_client: Any | None = None  # Injected dependency
    name: str = "cluster_web_search"
    weight: float = 83.0  # ~83s actual

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            cluster_plans = ctx.get_extra("cluster_search_plans", []) or []
            if not cluster_plans:
                Trace.event("retrieval.cluster_search.skipped", {"reason": "no_plans"})
                return ctx

            profile_name = resolve_profile_name(ctx.mode.name)
            profile = default_search_policy().get_profile(profile_name)
            search_depth = profile.search_depth or "basic"
            max_results = int(profile.max_results or 5)

            runtime = getattr(self.config, "runtime", None)
            deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())

            url_metadata: dict[str, dict[str, Any]] = {}
            url_variants: dict[str, set[str]] = {}
            cluster_url_map: dict[str, list[str]] = {}
            cluster_sufficiency: dict[str, float] = {}
            execution_states: dict[str, ClaimExecutionState] = ctx.get_extra("execution_states", {}) or {}

            # Semaphore to bound concurrent Tavily API calls (rate limit safety)
            runtime_config = ctx.get_extra("runtime_config")
            max_search_concurrency = 4  # default
            if runtime_config and hasattr(runtime_config, "llm"):
                max_search_concurrency = getattr(runtime_config.llm, "max_doc_concurrency", 4)
            max_search_concurrency = _coerce_int(max_search_concurrency, default=4, min_v=1, max_v=16)
            search_sem = asyncio.Semaphore(max_search_concurrency)

            # Stage 0: Aggregate and deduplicate unique queries across all clusters
            all_queries_to_run = []
            query_to_clusters: dict[str, list[str]] = {}
            for plan in cluster_plans:
                cluster_id = str(plan.get("cluster_id") or "cluster")
                queries = plan.get("search_queries") or []
                for q in queries:
                    if q:
                        all_queries_to_run.append(q)
                        query_to_clusters.setdefault(q, []).append(cluster_id)
            
            # Unique queries across global scope
            unique_queries = sorted(list(set(all_queries_to_run)))

            async def _search_with_sem(query: str, **kwargs) -> tuple[Any, list]:
                """Run a single search_phase call bounded by semaphore."""
                async with search_sem:
                    return await self.search_mgr.search_phase(query, **kwargs)

            async def _execute_single_query_search(query: str, claims_for_query: list[dict]):
                # Run trusted + general search in PARALLEL (instead of serial fallback)
                trusted_domains = get_trusted_domains_by_lang(ctx.lang or "en")

                trusted_task = _search_with_sem(
                    query,
                    max_results=max_results,
                    depth=search_depth,
                    topic="general",
                    include_domains=trusted_domains,
                )
                general_task = _search_with_sem(
                    query,
                    max_results=max_results,
                    depth=search_depth,
                    topic="general",
                    exclude_domains=trusted_domains,
                )

                (_, trusted_sources), (_, general_sources) = await asyncio.gather(
                    trusted_task, general_task
                )

                # Merge: trusted first, then general
                sources = (trusted_sources or []) + (general_sources or [])

                rep_claim = claims_for_query[0] if claims_for_query else {"text": query}
                sufficiency = check_sufficiency_for_claim(rep_claim, sources)

                Trace.event("retrieval.cluster_search.parallel_search", {
                    "query": query,
                    "trusted_count": len(trusted_sources or []),
                    "general_count": len(general_sources or []),
                })

                # Academic search only for SCIENTIFIC claims + insufficient evidence
                is_scientific = any(
                    c.get("policy_mode") == "SCIENTIFIC" 
                    or c.get("search_method") == "academic"
                    for c in claims_for_query
                )
                if is_scientific and sufficiency.status != SufficiencyStatus.SUFFICIENT:
                    Trace.event("retrieval.cluster_search.escalation", {
                        "query": query,
                        "reason": "bayesian_insufficient_scientific",
                        "confidence": sufficiency.reason,
                    })
                    # Tavily accepts only topic: general | news | finance
                    _, academic_sources = await _search_with_sem(
                        query,
                        max_results=max_results,
                        depth="advanced",
                        topic="general",
                    )
                    if academic_sources:
                        sources = sources + academic_sources
                        # Re-check sufficiency with academic results
                        sufficiency = check_sufficiency_for_claim(rep_claim, sources)
                
                return query, sources, sufficiency

            # Step 1: Run all UNIQUE queries in parallel
            query_to_rep_claims: dict[str, list[dict]] = {}
            for plan in cluster_plans:
                queries = plan.get("search_queries") or []
                claims = plan.get("claims") or []
                for q in queries:
                    if q:
                        query_to_rep_claims.setdefault(q, []).extend(claims)

            search_phase_started = time.perf_counter()
            Trace.event(
                "retrieval.cluster_search.phase.search.start",
                {
                    "queries": len(unique_queries),
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "max_parallel": max_search_concurrency,
                },
            )
            search_results = await asyncio.gather(*[
                _execute_single_query_search(q, query_to_rep_claims.get(q, []))
                for q in unique_queries
            ])
            Trace.event(
                "retrieval.cluster_search.phase.search.complete",
                {
                    "queries": len(unique_queries),
                    "duration_ms": int((time.perf_counter() - search_phase_started) * 1000),
                },
            )
            
            # Map results back to queries
            query_to_sources = {q: (s, suf) for q, s, suf in search_results}

            # Step 2: Assemble results for each cluster
            for plan in cluster_plans:
                cluster_id = str(plan.get("cluster_id") or "cluster")
                queries = plan.get("search_queries") or []
                cluster_urls: list[str] = []
                claims_list = plan.get("claims") or []
                query_origin = plan.get("query_origin") or "planned"
                fallback_reason = plan.get("fallback_reason")

                for hop_index, query in enumerate(queries):
                    if not query:
                        continue
                    
                    sources, sufficiency = query_to_sources.get(query, ([], None))
                    if sufficiency is None: # Should not happen
                        rep_claim = claims_list[0] if claims_list else {"text": query}
                        sufficiency = check_sufficiency_for_claim(rep_claim, sources)

                    # Record sufficiency and hops
                    current_p = cluster_sufficiency.get(cluster_id, 0.0)
                    cluster_sufficiency[cluster_id] = max(current_p, sufficiency.posterior_p)

                    retrieval_eval = {"query_origin": query_origin}
                    if fallback_reason:
                        retrieval_eval["fallback_reason"] = fallback_reason

                    hop = RetrievalHop(
                        hop_index=hop_index + 1,
                        query=query,
                        decision=sufficiency.status,
                        reason=sufficiency.reason,
                        phase_id="deep_v2_cluster_search",
                        query_type=search_depth,
                        results_count=len(sources),
                        retrieval_eval=retrieval_eval,
                    )
                    
                    for claim in claims_list:
                        cid = claim.get("id") or claim.get("claim_id")
                        if not cid:
                            continue
                        if cid not in execution_states:
                            execution_states[cid] = ClaimExecutionState(claim_id=cid)
                        execution_states[cid].hops.append(hop)
                        execution_states[cid].mark_completed("deep_v2_cluster_search")

                    # Collect URLs
                    for source in sources:
                        if not isinstance(source, dict):
                            continue
                        raw_url = source.get("url") or source.get("link")
                        if not raw_url:
                            continue
                        canonical = normalize_url(str(raw_url))
                        if not canonical:
                            continue
                        url_variants.setdefault(canonical, set()).add(str(raw_url))
                        if canonical not in url_metadata:
                            score = source.get("score")
                            if score is None:
                                score = source.get("provider_score")
                            if score is None:
                                score = source.get("relevance_score")
                            url_metadata[canonical] = {
                                "url": canonical,
                                "title": source.get("title") or "",
                                "snippet": source.get("content") or source.get("snippet") or "",
                                "score": _coerce_score(score),
                                "source_id": source.get("source_id") or source_id_for_url(canonical),
                                "provider_meta": dict(source),
                            }
                        if canonical not in cluster_urls:
                            cluster_urls.append(canonical)
                
                cluster_url_map[cluster_id] = cluster_urls

            total_queries = len(unique_queries)

            unique_urls = sorted(url_metadata.keys())
            urls_before_cap = len(unique_urls)
            max_unique_urls = _coerce_int(
                getattr(deep_v2_cfg, "max_unique_urls", 120),
                default=120,
                min_v=0,
                max_v=500,
            )
            if max_unique_urls > 0 and len(unique_urls) > max_unique_urls:
                unique_urls = sorted(
                    unique_urls,
                    key=lambda url: (
                        -_coerce_score((url_metadata.get(url) or {}).get("score")),
                        url,
                    ),
                )[:max_unique_urls]
                Trace.event(
                    "retrieval.cluster_search.url_cap.applied",
                    {
                        "urls_before": urls_before_cap,
                        "urls_after": len(unique_urls),
                        "urls_dropped": urls_before_cap - len(unique_urls),
                        "max_unique_urls": max_unique_urls,
                    },
                )

            extract_phase_started = time.perf_counter()
            Trace.event(
                "retrieval.cluster_search.phase.extract.start",
                {
                    "urls_before_cap": urls_before_cap,
                    "urls_after_cap": len(unique_urls),
                    "max_unique_urls": max_unique_urls,
                },
            )
            content_map = await self.search_mgr.fetch_urls_content_batch(unique_urls, stage=None)
            Trace.event(
                "retrieval.cluster_search.phase.extract.complete",
                {
                    "urls_after_cap": len(unique_urls),
                    "fetched_urls": len(content_map),
                    "duration_ms": int((time.perf_counter() - extract_phase_started) * 1000),
                },
            )

            evidence_docs: dict[str, dict[str, Any]] = {}
            ordered_texts: list[str] = []
            ordered_urls: list[str] = []
            for url in unique_urls:
                text = content_map.get(url) or ""
                ordered_urls.append(url)
                ordered_texts.append(str(text))

            # Rely on injected EmbeddingClient
            # If not provided, skip clustering (each URL is its own cluster)
            cluster_ids: dict[str, str] = {}
            embeddings: list[list[float]] = []

            if self.embedding_client:
                # Document embeddings must not use full page blobs. Use a bounded excerpt.
                # This is a semantic 'document' embedding, not corpus indexing.
                ordered_embed_texts: list[str] = []
                for t in ordered_texts:
                    s = str(t)
                    # Prefer early part; heavy pages often append nav/related content later.
                    ordered_embed_texts.append(s[:8000])

                embed_phase_started = time.perf_counter()
                Trace.event(
                    "retrieval.cluster_search.phase.embedding.start",
                    {
                        "documents": len(ordered_embed_texts),
                        "max_chars_per_doc": 8000,
                    },
                )
                embeddings = await self.embedding_client.embed_texts(ordered_embed_texts, purpose="document")
                sim_matrix = self.embedding_client.build_similarity_matrix(embeddings)
                
                # Use shared logic from use case
                cluster_ids = assign_similarity_clusters(
                    ordered_urls,
                    sim_matrix,
                    quantile=deep_v2_cfg.doc_cluster_quantile,
                )
                Trace.event(
                    "retrieval.cluster_search.phase.embedding.complete",
                    {
                        "documents": len(ordered_embed_texts),
                        "duration_ms": int((time.perf_counter() - embed_phase_started) * 1000),
                    },
                )
            else:
                Trace.event(
                    "retrieval.cluster_search.phase.embedding.skipped",
                    {"reason": "embedding_client_unavailable", "documents": len(ordered_urls)},
                )

            for idx, url in enumerate(ordered_urls):
                cleaned_text = ordered_texts[idx]
                meta = url_metadata.get(url, {})
                publisher_id = get_registrable_domain(url) or ""
                content_hash = (
                    hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
                    if cleaned_text
                    else ""
                )
                evidence_docs[url] = {
                    "canonical_url": url,
                    "url_variants": sorted(url_variants.get(url, {url})),
                    "cleaned_text": cleaned_text,
                    "content_hash": content_hash,
                    "publisher_id": publisher_id,
                    "embedding": embeddings[idx] if idx < len(embeddings) else None,
                    "similar_cluster_id": cluster_ids.get(url) or _stable_cluster_id([url]),
                    "title": meta.get("title") or "",
                    "snippet": meta.get("snippet") or "",
                    "provider_score": meta.get("score"),
                    "source_id": meta.get("source_id"),
                    "provider_meta": dict(meta.get("provider_meta") or {}),
                }

            cluster_evidence_docs: dict[str, list[dict[str, Any]]] = {}
            for cluster_id, urls in cluster_url_map.items():
                cluster_evidence_docs[cluster_id] = [evidence_docs[u] for u in urls if u in evidence_docs]

            plan = ctx.get_extra(SEARCH_PLAN_KEY)
            plan_id = getattr(plan, "plan_id", None)

            Trace.event(
                "retrieval.cluster_search.completed",
                {
                    "plan_id": plan_id,
                    "clusters": len(cluster_evidence_docs),
                    "urls_total": len(evidence_docs),
                    "urls_before_cap": urls_before_cap,
                    "url_cap": max_unique_urls,
                },
            )

            evidence_doc_meta = {}
            for url, doc in evidence_docs.items():
                evidence_doc_meta[url] = {
                    "content_hash": doc.get("content_hash"),
                    "publisher_id": doc.get("publisher_id"),
                    "similar_cluster_id": doc.get("similar_cluster_id"),
                    "canonical_url": doc.get("canonical_url") or url,
                }

            return (
                ctx.set_extra("cluster_evidence_docs", cluster_evidence_docs)
                .set_extra("evidence_docs", evidence_docs)
                .set_extra("evidence_doc_meta", evidence_doc_meta)
                .set_extra("cluster_sufficiency", cluster_sufficiency)
                .set_extra("execution_states", execution_states)
                .set_extra(
                    "retrieval_search_trace",
                    {
                        "plan_id": plan_id,
                        "clusters": len(cluster_evidence_docs),
                        "queries": total_queries,
                        "urls_total": len(evidence_docs),
                        "urls_before_cap": urls_before_cap,
                        "url_cap": max_unique_urls,
                    },
                )
            )
        except Exception as exc:
            logger.exception("[ClusterWebSearchStep] Failed: %s", exc)
            raise PipelineExecutionError(self.name, str(exc), cause=exc) from exc
