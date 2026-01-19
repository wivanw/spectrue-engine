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

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.graph.embedding_util import EmbeddingClient
from spectrue_core.pipeline.contracts import SEARCH_PLAN_KEY
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.tools.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.utils.trace import Trace
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.verification.orchestration.sufficiency import check_sufficiency_for_claim, SufficiencyStatus
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url, source_id_for_url
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)

logger = logging.getLogger(__name__)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return score


def _stable_cluster_id(urls: list[str]) -> str:
    ordered = sorted([u for u in urls if u])
    raw = "|".join(ordered)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    idx = int(round(q * (len(ordered) - 1)))
    return float(ordered[idx])


def _assign_similarity_clusters(
    urls: list[str],
    sim_matrix: list[list[float]],
    *,
    quantile: float,
) -> dict[str, str]:
    if len(urls) <= 1:
        return {urls[0]: _stable_cluster_id(urls)} if urls else {}

    sims: list[float] = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sims.append(float(sim_matrix[i][j]))
    tau = _quantile(sims, quantile)

    adjacency: dict[str, set[str]] = {u: set() for u in urls}
    for i, src in enumerate(urls):
        for j, dst in enumerate(urls):
            if i >= j:
                continue
            if float(sim_matrix[i][j]) >= tau:
                adjacency[src].add(dst)
                adjacency[dst].add(src)

    visited: set[str] = set()
    clusters: dict[str, str] = {}
    for url in urls:
        if url in visited:
            continue
        queue = [url]
        visited.add(url)
        component: list[str] = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        cluster_id = _stable_cluster_id(component)
        for item in component:
            clusters[item] = cluster_id

    return clusters


@dataclass
class ClusterWebSearchStep:
    """Execute cluster-level search/extract for deep_v2."""

    config: Any
    search_mgr: Any
    name: str = "cluster_web_search"

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

            total_queries = 0
            for plan in cluster_plans:
                cluster_id = str(plan.get("cluster_id") or "cluster")
                queries = plan.get("search_queries") or []
                cluster_urls: list[str] = []
                for query in queries:
                    if not query:
                        continue
                    total_queries += 1
                    # Stage 1: Trusted (Tier 1) search
                    trusted_domains = get_trusted_domains_by_lang(ctx.lang or "en")
                    _, sources = await self.search_mgr.search_phase(
                        query,
                        max_results=max_results,
                        depth=search_depth,
                        topic="general",
                        include_domains=trusted_domains,
                    )

                    # Fallback logic: if trusted results are poor (Bayesian assessment), try general search
                    # For cluster search, we use the first claim of the cluster as a representative for sufficiency
                    claims_list = plan.get("claims") or []
                    rep_claim = claims_list[0] if claims_list else {"text": query}
                    sufficiency = check_sufficiency_for_claim(rep_claim, sources or [])
                    
                    if sufficiency.status != SufficiencyStatus.SUFFICIENT:
                        Trace.event("retrieval.cluster_search.fallback", {
                            "query": query,
                            "reason": "bayesian_insufficient",
                            "confidence": sufficiency.reason,
                        })
                        _, general_sources = await self.search_mgr.search_phase(
                            query,
                            max_results=max_results,
                            depth=search_depth,
                            topic="general",
                            exclude_domains=trusted_domains,
                        )
                        if general_sources:
                            # Combine or prefer general results if they are better
                            # For simplicity, we just add them to the sources list
                            sources = (sources or []) + general_sources

                    # Final Bayesian sufficiency after fallback
                    final_sufficiency = check_sufficiency_for_claim(rep_claim, sources or [])
                    current_p = cluster_sufficiency.get(cluster_id, 0.0)
                    cluster_sufficiency[cluster_id] = max(current_p, final_sufficiency.posterior_p)

                    for source in sources or []:
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

            unique_urls = sorted(url_metadata.keys())
            content_map = await self.search_mgr.fetch_urls_content_batch(unique_urls, stage=None)

            evidence_docs: dict[str, dict[str, Any]] = {}
            ordered_texts: list[str] = []
            ordered_urls: list[str] = []
            for url in unique_urls:
                text = content_map.get(url) or ""
                ordered_urls.append(url)
                ordered_texts.append(str(text))

            embedding_client = EmbeddingClient()
            # Document embeddings must not use full page blobs. Use a bounded excerpt.
            # This is a semantic 'document' embedding, not corpus indexing.
            ordered_embed_texts: list[str] = []
            for t in ordered_texts:
                s = str(t)
                # Prefer early part; heavy pages often append nav/related content later.
                ordered_embed_texts.append(s[:8000])
            embeddings = await embedding_client.embed_texts(ordered_embed_texts, purpose="document")
            sim_matrix = embedding_client.build_similarity_matrix(embeddings)
            cluster_ids = _assign_similarity_clusters(
                ordered_urls,
                sim_matrix,
                quantile=deep_v2_cfg.doc_cluster_quantile,
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
                .set_extra(
                    "retrieval_search_trace",
                    {
                        "plan_id": plan_id,
                        "clusters": len(cluster_evidence_docs),
                        "queries": total_queries,
                        "urls_total": len(evidence_docs),
                    },
                )
            )
        except Exception as exc:
            logger.exception("[ClusterWebSearchStep] Failed: %s", exc)
            raise PipelineExecutionError(self.name, str(exc), cause=exc) from exc
