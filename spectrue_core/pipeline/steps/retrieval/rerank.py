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
    RANKED_RESULTS_KEY,
    RankedResultItem,
    RankedResults,
    RawSearchResults,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)

logger = logging.getLogger(__name__)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, score))


def _rank_items(items: list[RankedResultItem], max_results: int | None) -> tuple[RankedResultItem, ...]:
    items.sort(key=lambda i: i.blended_score, reverse=True)
    if max_results and max_results > 0:
        items = items[:max_results]
    return tuple(items)


@dataclass
class RerankStep:
    """Blend provider and similarity scores without filtering."""

    name: str = "rerank_results"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            raw_results: RawSearchResults | None = ctx.get_extra(RAW_SEARCH_RESULTS_KEY)
            plan_id = raw_results.plan_id if raw_results else "plan-unknown"

            profile_name = resolve_profile_name(ctx.search_type)
            profile = default_search_policy().get_profile(profile_name)
            rerank_lambda = float(profile.quality_thresholds.rerank_lambda)
            max_results = int(profile.max_results or 0)

            by_claim_items: dict[str, list[RankedResultItem]] = {}
            global_items: list[RankedResultItem] = []

            sources_iter = []
            if raw_results:
                sources_iter = list(raw_results.results)
            elif ctx.sources:
                sources_iter = [
                    {
                        "claim_id": None,
                        "provider_payload": ctx.sources,
                    }
                ]

            for entry in sources_iter:
                if isinstance(entry, dict):
                    claim_id = entry.get("claim_id")
                    payload = entry.get("provider_payload", [])
                    query_id = "legacy"
                else:
                    claim_id = entry.claim_id
                    payload = entry.provider_payload
                    query_id = entry.query_id

                if not isinstance(payload, list):
                    continue

                for src in payload:
                    if not isinstance(src, dict):
                        continue
                    url = src.get("url") or src.get("link")
                    if not url:
                        continue
                    provider_score = _coerce_score(
                        src.get("provider_score")
                        if src.get("provider_score") is not None
                        else src.get("score")
                    )
                    similarity_score = _coerce_score(
                        src.get("similarity_score")
                        if src.get("similarity_score") is not None
                        else src.get("sim")
                    )
                    if similarity_score == 0.0:
                        similarity_score = _coerce_score(src.get("relevance_score"))
                    if similarity_score == 0.0:
                        similarity_score = provider_score

                    blended = rerank_lambda * provider_score + (1 - rerank_lambda) * similarity_score

                    item = RankedResultItem(
                        url=str(url),
                        provider_score=provider_score,
                        similarity_score=similarity_score,
                        blended_score=blended,
                        claim_id=str(claim_id) if claim_id else src.get("claim_id"),
                        title=src.get("title"),
                        snippet=src.get("snippet") or src.get("content"),
                        raw=dict(src, _query_id=query_id),
                    )

                    if item.claim_id:
                        by_claim_items.setdefault(str(item.claim_id), []).append(item)
                        if not ctx.mode.allow_batch:
                            global_items.append(item)
                    else:
                        global_items.append(item)

            results_by_claim = {
                claim_id: _rank_items(items, max_results)
                for claim_id, items in by_claim_items.items()
            }
            results_global = _rank_items(global_items, max_results)

            ranked = RankedResults(
                plan_id=plan_id,
                results_global=results_global,
                results_by_claim=results_by_claim,
                trace={
                    "rerank_lambda": rerank_lambda,
                    "max_results": max_results,
                    "claims": len(results_by_claim),
                    "global_count": len(results_global),
                },
            )

            Trace.event(
                "retrieval.rerank",
                {
                    "plan_id": plan_id,
                    "rerank_lambda": rerank_lambda,
                    "claims": len(results_by_claim),
                    "global": len(results_global),
                },
            )

            return (
                ctx.set_extra(RANKED_RESULTS_KEY, ranked)
                .set_extra(
                    "retrieval_rerank_trace",
                    {
                        "plan_id": plan_id,
                        "claims": len(results_by_claim),
                        "global": len(results_global),
                    },
                )
            )

        except Exception as e:
            logger.exception("[RerankStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
