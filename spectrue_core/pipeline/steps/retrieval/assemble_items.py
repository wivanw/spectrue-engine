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
    RANKED_RESULTS_KEY,
    RETRIEVAL_ITEMS_KEY,
    RankedResults,
    RetrievalItem,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.fixed_pipeline import source_id_for_url

logger = logging.getLogger(__name__)


def _build_item_payload(item: RetrievalItem) -> dict[str, Any]:
    payload = item.to_payload()
    if payload.get("similarity_score") is not None:
        payload["sim"] = payload.get("similarity_score")
    if payload.get("content_excerpt"):
        payload["content"] = payload.get("content_excerpt")
    return payload


def _merge_enriched(item_raw: dict[str, Any], enriched: dict[str, Any]) -> dict[str, Any]:
    merged = dict(item_raw)
    for key in ("title", "snippet", "quote", "content", "content_excerpt"):
        val = enriched.get(key)
        if val:
            merged[key] = val
    return merged


@dataclass
class AssembleRetrievalItemsStep:
    """Assemble retrieval items from ranked results and optional chunks."""

    name: str = "assemble_retrieval_items"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ranked: RankedResults | None = ctx.get_extra(RANKED_RESULTS_KEY)
            if ranked is None:
                return ctx

            chunks_by_url: dict[str, dict[str, Any]] = ctx.get_extra("chunks_by_url", {}) or {}

            global_items: list[RetrievalItem] = []
            by_claim_items: dict[str, list[RetrievalItem]] = {}
            source_payloads: list[dict[str, Any]] = []

            for idx, item in enumerate(ranked.results_global, start=1):
                enriched = chunks_by_url.get(item.url, {})
                merged = _merge_enriched(item.raw, enriched)
                source_id = merged.get("source_id") or item.raw.get("source_id") or source_id_for_url(item.url)
                retrieval_item = RetrievalItem(
                    url=item.url,
                    source_id=source_id,
                    provider_score=item.provider_score,
                    similarity_score=item.similarity_score,
                    blended_score=item.blended_score,
                    claim_id=item.claim_id,
                    title=merged.get("title") or item.title,
                    snippet=merged.get("snippet") or item.snippet,
                    quote=merged.get("quote"),
                    content_excerpt=merged.get("content") or merged.get("content_excerpt"),
                    rank=idx,
                )
                global_items.append(retrieval_item)
                source_payloads.append(_build_item_payload(retrieval_item))

            for claim_id, items in ranked.results_by_claim.items():
                by_claim_items[claim_id] = []
                for idx, item in enumerate(items, start=1):
                    enriched = chunks_by_url.get(item.url, {})
                    merged = _merge_enriched(item.raw, enriched)
                    source_id = merged.get("source_id") or item.raw.get("source_id") or source_id_for_url(item.url)
                    retrieval_item = RetrievalItem(
                        url=item.url,
                        source_id=source_id,
                        provider_score=item.provider_score,
                        similarity_score=item.similarity_score,
                        blended_score=item.blended_score,
                        claim_id=claim_id,
                        title=merged.get("title") or item.title,
                        snippet=merged.get("snippet") or item.snippet,
                        quote=merged.get("quote"),
                        content_excerpt=merged.get("content") or merged.get("content_excerpt"),
                        rank=idx,
                    )
                    by_claim_items[claim_id].append(retrieval_item)
                    source_payloads.append(_build_item_payload(retrieval_item))

            Trace.event(
                "retrieval.items",
                {
                    "plan_id": ranked.plan_id,
                    "global": len(global_items),
                    "claims": len(by_claim_items),
                },
            )

            return (
                ctx.with_update(sources=source_payloads)
                .set_extra(
                    RETRIEVAL_ITEMS_KEY,
                    {
                        "global": tuple(global_items),
                        "by_claim": {cid: tuple(items) for cid, items in by_claim_items.items()},
                    },
                )
            )

        except Exception as e:
            logger.exception("[AssembleRetrievalItemsStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
