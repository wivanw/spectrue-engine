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

"""Data helpers for `WebSearchStep` (fixed-pipeline web search).

Pure bookkeeping over search results and extracted content; no I/O.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.pipeline.contracts import SearchPlan
from spectrue_core.pipeline.retrieval.cegs_mvp import (
    EvidenceItem,
    EvidencePool,
    EvidenceSourceMeta,
    _compute_content_hash,
)
from spectrue_core.utils.retrieval_urls import (
    ExtractedContent,
    normalize_url,
    source_id_for_url,
)


def claim_id_for(claim: dict[str, Any], idx: int) -> str:
    raw = claim.get("id") or claim.get("claim_id")
    if raw:
        return str(raw)
    return f"c{idx + 1}"


def coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return score


def query_dedup_key(query: str) -> str:
    return " ".join(query.lower().split())


def first_query_for_claim(
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


def record_sources(
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
                "score": coerce_score(score),
                "source_id": source_id,
                "provider_meta": dict(source, source_id=source_id),
            }
    return urls


def apply_metadata(
    extracted: dict[str, ExtractedContent],
    url_metadata: dict[str, dict[str, Any]],
) -> None:
    for url, content in extracted.items():
        meta = url_metadata.get(url)
        if meta:
            content.metadata.update(meta)
        content.metadata.setdefault("url", url)
        content.metadata.setdefault("source_id", meta.get("source_id") if meta else source_id_for_url(url))


def build_evidence_pool(
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
            score=coerce_score(meta.get("score")),
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


def build_sources_for_urls(
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
        if meta.get("decision_evidence"):
            src["decision_evidence"] = dict(meta["decision_evidence"])  # G3 readings, reused by G1
        sources.append(src)
    return sources
