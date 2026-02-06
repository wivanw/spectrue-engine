# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Evidence clustering helpers."""

from __future__ import annotations
import logging
from typing import Any

from spectrue_core.domain.evidence.evidence_pack import SearchResult

logger = logging.getLogger(__name__)

HIGH_TIER_SET = {"A", "A'", "B"}
LOW_TIER_SET = {"C", "D"}
TIER_RANK = {"D": 1, "C": 2, "B": 3, "A'": 4, "A": 4}


def group_by_similar_cluster_id(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group evidence items by their precomputed similar_cluster_id."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("similar_cluster_id") or "")
        if not cid:
            continue
        clusters.setdefault(cid, []).append(item)
    return clusters


def _normalize_tier(*, tier_raw: str | None, source_type: str | None) -> str:
    if tier_raw:
        return str(tier_raw).strip().upper()
    stype = (source_type or "").strip().lower()
    match stype:
        case "primary":
            return "A"
        case "official":
            return "A'"
        case "independent_media":
            return "B"
        case "social":
            return "D"
        case _:
            return "C"


def _tier_rank(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def merge_stance_passes(
    *,
    support_results: list[SearchResult],
    refute_results: list[SearchResult],
    original_sources: list[dict],
) -> list[SearchResult]:
    """
    Merge results from support and refute stance detection passes.
    Chooses the stance with higher tier/confidence/rank.
    """
    if not support_results:
        return refute_results
    if not refute_results:
        return support_results
    if len(support_results) != len(refute_results):
        logger.warning(
            "[StanceMerge] Pass count mismatch: support=%d refute=%d",
            len(support_results),
            len(refute_results),
        )
        return support_results

    merged: list[SearchResult] = []
    for idx, support in enumerate(support_results):
        refute = refute_results[idx]
        src = original_sources[idx] if idx < len(original_sources) else {}

        support_quote = support.get("quote_span") or support.get("key_snippet")
        refute_quote = refute.get("contradiction_span") or refute.get("key_snippet")

        support_tier = _normalize_tier(
            tier_raw=support.get("evidence_tier") or src.get("evidence_tier"),
            source_type=support.get("source_type") or src.get("source_type"),
        )
        refute_tier = _normalize_tier(
            tier_raw=refute.get("evidence_tier") or src.get("evidence_tier"),
            source_type=refute.get("source_type") or src.get("source_type"),
        )

        support_rank = _tier_rank(support_tier)
        refute_rank = _tier_rank(refute_tier)

        support_has = (support.get("stance") == "support") and bool(support_quote)
        refute_has = (refute.get("stance") == "refute") and bool(refute_quote)

        if refute_has and refute_rank >= support_rank:
            merged_result = dict(refute)
            merged_result["stance"] = "refute"
            merged_result["pass_type"] = "REFUTE_ONLY"
            merged_result["contradiction_span"] = refute_quote
            merged_result["evidence_tier"] = refute_tier
        elif support_has:
            merged_result = dict(support)
            merged_result["stance"] = "support"
            merged_result["pass_type"] = "SUPPORT_ONLY"
            merged_result["quote_span"] = support_quote
            merged_result["evidence_tier"] = support_tier
            if support_tier in LOW_TIER_SET:
                merged_result["stance_confidence"] = "low"
        elif refute_has:
            merged_result = dict(refute)
            merged_result["stance"] = "refute"
            merged_result["pass_type"] = "REFUTE_ONLY"
            merged_result["contradiction_span"] = refute_quote
            merged_result["evidence_tier"] = refute_tier
        else:
            merged_result = dict(support)
            merged_result["stance"] = "context"
            merged_result["pass_type"] = "SUPPORT_ONLY"

        if merged_result.get("stance") not in ("support", "refute"):
            merged_result["quote_span"] = None
            merged_result["contradiction_span"] = None

        url = merged_result.get("url") or src.get("url") or src.get("link")
        if url:
            merged_result["evidence_refs"] = [url]

        merged.append(merged_result)  # type: ignore

    return merged
