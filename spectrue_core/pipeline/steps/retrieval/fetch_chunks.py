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

from spectrue_core.pipeline.contracts import RANKED_RESULTS_KEY, RankedResults
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def _claim_text_map(claims: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        cid = claim.get("id") or claim.get("claim_id")
        if not cid:
            continue
        text = claim.get("normalized_text") or claim.get("text") or ""
        mapping[str(cid)] = text
    return mapping


def _prepare_sources(items, claim_text: str | None) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in items:
        src = dict(item.raw) if isinstance(item.raw, dict) else {}
        src.setdefault("url", item.url)
        if claim_text:
            src.setdefault("claim_text", claim_text)
        sources.append(src)
    return sources


@dataclass
class FetchChunksStep:
    """Optionally enrich ranked results with content and quotes."""

    search_mgr: Any
    name: str = "fetch_chunks"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ranked: RankedResults | None = ctx.get_extra(RANKED_RESULTS_KEY)
            if ranked is None:
                return ctx

            claim_texts = _claim_text_map(ctx.claims)
            chunks_by_url: dict[str, dict[str, Any]] = {}

            # === PHASE 0: Aggregate all URLs for batch pre-fetch ===
            # This ensures all URLs are fetched in one API call (5 URLs = 1 credit)
            # Then individual EAL calls get cache hits instead of API calls
            all_urls: list[str] = []
            
            for claim_id, items in ranked.results_by_claim.items():
                for item in items:
                    url = getattr(item, "url", None) or (item.raw.get("url") if isinstance(item.raw, dict) else None)
                    if url and url not in all_urls:
                        all_urls.append(url)
            
            if ranked.results_global:
                for item in ranked.results_global:
                    url = getattr(item, "url", None) or (item.raw.get("url") if isinstance(item.raw, dict) else None)
                    if url and url not in all_urls:
                        all_urls.append(url)
            
            # Pre-fetch all URLs in batches of 5
            if all_urls and hasattr(self.search_mgr, "fetch_urls_content_batch"):
                Trace.event("fetch_chunks.prefetch.start", {
                    "total_urls": len(all_urls),
                })
                try:
                    # This populates the cache; EAL will get cache hits
                    await self.search_mgr.fetch_urls_content_batch(all_urls)
                    Trace.event("fetch_chunks.prefetch.done", {
                        "urls_prefetched": len(all_urls),
                    })
                except Exception as e:
                    logger.warning("[FetchChunksStep] Prefetch failed: %s", e)
                    Trace.event("fetch_chunks.prefetch.error", {"error": str(e)[:200]})

            # === PHASE 1: Per-claim EAL (now gets cache hits) ===
            for claim_id, items in ranked.results_by_claim.items():
                claim_text = claim_texts.get(claim_id)
                sources = _prepare_sources(items, claim_text)
                if not sources:
                    continue
                enriched = await self.search_mgr.apply_evidence_acquisition_ladder(
                    sources,
                    budget_context="claim",
                    claim_id=claim_id,
                    cache_only=True,  # Use pre-fetched cache from PHASE 0
                )
                for src in enriched:
                    url = src.get("url") or src.get("link")
                    if url:
                        chunks_by_url[str(url)] = src

            if ranked.results_global:
                sources = _prepare_sources(ranked.results_global, None)
                if sources:
                    enriched = await self.search_mgr.apply_evidence_acquisition_ladder(
                        sources,
                        budget_context="claim",
                        cache_only=True,  # Use pre-fetched cache from PHASE 0
                    )
                    for src in enriched:
                        url = src.get("url") or src.get("link")
                        if url and url not in chunks_by_url:
                            chunks_by_url[str(url)] = src

            Trace.event(
                "retrieval.fetch_chunks",
                {
                    "plan_id": ranked.plan_id,
                    "chunks": len(chunks_by_url),
                },
            )

            return ctx.set_extra("chunks_by_url", chunks_by_url)

        except Exception as e:
            logger.exception("[FetchChunksStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
