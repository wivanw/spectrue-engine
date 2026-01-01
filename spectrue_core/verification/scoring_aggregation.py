# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Deterministic claim verdict aggregation.

Implements evidence-quoted, conflict-aware scoring with penalties
for temporal mismatch and consistency gaps.

.. deprecated:: M104
    This module is deprecated. Use :mod:`spectrue_core.scoring.belief` for
    Bayesian scoring instead. This module will be removed in a future version.
    See FR-010: All arithmetic averaging must be removed from the scoring pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any
from spectrue_core.utils.trace import Trace

# Deprecation warning at module import time (only once)
warnings.warn(
    "spectrue_core.verification.scoring_aggregation is deprecated. "
    "Migrate to spectrue_core.scoring.belief (Bayesian scoring). "
    "See FR-010.",
    DeprecationWarning,
    stacklevel=2,
)

TIER_RANK = {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}

AMBIGUOUS_MIN = 0.35
AMBIGUOUS_MAX = 0.65


def _tier_rank(tier: str | None) -> int:
    if not tier:
        return 0
    return TIER_RANK.get(str(tier).strip().upper(), 0)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _best_evidence(items: list[dict], *, stance: str) -> dict | None:
    best: dict | None = None
    best_score = -1.0
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("stance") or "").upper() != stance:
            continue
        if not item.get("quote"):
            continue
        relevance = item.get("relevance")
        try:
            score = float(relevance) if relevance is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        if score > best_score:
            best_score = score
            best = item
    return best


def _score_from_quote_counts(*, support_count: int, refute_count: int) -> tuple[str, float]:
    if support_count > 0 and refute_count == 0:
        base = 0.75 if support_count == 1 else 0.85
        return "verified", base
    if refute_count > 0 and support_count == 0:
        base = 0.25 if refute_count == 1 else 0.15
        return "refuted", base
    return "ambiguous", 0.5


def aggregate_claim_verdict(
    evidence_pack: dict,
    *,
    policy: dict | None = None,
    claim_id: str | None = None,
    temporality: dict | None = None,
) -> dict[str, Any]:
    """
    Aggregate a claim verdict from evidence items and policy constraints.

    Returns dict with verdict, verdict_score, and reasons_expert details.
    """
    policy = policy or {}
    items = evidence_pack.get("items", []) or []
    if claim_id:
        items = [i for i in items if i.get("claim_id") in (None, claim_id) or i.get("claim_id") == claim_id]

    support_best = _best_evidence(items, stance="SUPPORT")
    refute_best = _best_evidence(items, stance="REFUTE")

    max_without_quotes = float(policy.get("max_confidence_without_quotes", 0.5))

    if not support_best and not refute_best:
        score = min(0.5, max_without_quotes)
        return {
            "verdict": "insufficient",
            "verdict_score": score,
            "reasons_expert": {
                "penalties": ["no_support_or_refute_quotes"],
            },
        }

    support_count = sum(
        1
        for i in items
        if isinstance(i, dict)
        and str(i.get("stance") or "").upper() == "SUPPORT"
        and i.get("quote")
    )
    refute_count = sum(
        1
        for i in items
        if isinstance(i, dict)
        and str(i.get("stance") or "").upper() == "REFUTE"
        and i.get("quote")
    )

    support_tier = support_best.get("tier") if support_best else None
    refute_tier = refute_best.get("tier") if refute_best else None

    conflict = bool(support_best and refute_best)
    verdict, base = _score_from_quote_counts(
        support_count=support_count,
        refute_count=refute_count,
    )

    core_count = support_count + refute_count
    side_count = sum(
        1
        for i in items
        if isinstance(i, dict)
        and i.get("quote")
        and str(i.get("stance") or "").upper() not in {"SUPPORT", "REFUTE"}
    )
    total_count = len([i for i in items if isinstance(i, dict)])

    conflict_weight = float(policy.get("penalty_conflict_weight", 0.10))
    temporal_weight = float(policy.get("penalty_temporal_weight", 0.15))
    diversity_weight = float(policy.get("penalty_diversity_weight", 0.05))

    conflict_mass = 0.0
    if core_count > 0:
        conflict_mass = min(support_count, refute_count) / max(1, core_count)
    conflict_penalty = conflict_weight * conflict_mass if conflict else 0.0
    temporal_mass = 0.0
    diversity_shortfall = 0.0

    penalties: list[str] = []
    if conflict:
        base -= conflict_penalty
        penalties.append("conflict_penalty")

    is_time_sensitive = bool((temporality or {}).get("is_time_sensitive"))
    forced_ambiguous = False
    if is_time_sensitive:
        outdated = any(str(i.get("temporal_flag") or "").lower() == "outdated" for i in items)
        if outdated:
            temporal_mass = sum(
                1
                for i in items
                if isinstance(i, dict)
                and str(i.get("temporal_flag") or "").lower() in {"outdated", "future"}
            ) / max(1, total_count)
            base -= temporal_weight * temporal_mass
            penalties.append("temporal_penalty")
            if verdict == "verified":
                verdict = "ambiguous"
                forced_ambiguous = True

    stats = evidence_pack.get("stats") or {}
    min_domain_div = int(policy.get("min_domain_diversity", 1))
    if isinstance(stats, dict) and stats.get("domain_diversity") is not None:
        domain_diversity = int(stats.get("domain_diversity") or 0)
        if domain_diversity < min_domain_div:
            diversity_shortfall = (min_domain_div - domain_diversity) / max(1, min_domain_div)
            base -= diversity_weight * diversity_shortfall
            penalties.append("consistency_penalty")

    best_tier = None
    if support_tier and refute_tier:
        best_tier = support_tier if _tier_rank(support_tier) >= _tier_rank(refute_tier) else refute_tier
    elif support_tier:
        best_tier = support_tier
    elif refute_tier:
        best_tier = refute_tier

    pre_ceiling = base
    base = _clamp(base, 0.0, 1.0)

    if conflict:
        verdict = "ambiguous"
        base = _clamp(base, AMBIGUOUS_MIN, AMBIGUOUS_MAX)
    if forced_ambiguous:
        base = _clamp(base, AMBIGUOUS_MIN, AMBIGUOUS_MAX)

    Trace.event(
        "verdict.tier_ceiling",
        {
            "claim_id": claim_id,
            "support_tier": support_tier,
            "refute_tier": refute_tier,
            "best_tier": best_tier,
            "pre_ceiling_score": pre_ceiling,
            "final_score": base,
            "conflict": conflict,
            "forced_ambiguous": forced_ambiguous,
            "support_quotes": support_count,
            "refute_quotes": refute_count,
            "core_quotes": core_count,
            "side_quotes": side_count,
            "penalty_weights": {
                "conflict": conflict_weight,
                "temporal": temporal_weight,
                "diversity": diversity_weight,
            },
            "penalty_masses": {
                "conflict_mass": conflict_mass,
                "temporal_mass": temporal_mass,
                "diversity_shortfall": diversity_shortfall,
            },
        },
    )

    return {
        "verdict": verdict,
        "verdict_score": base,
        "best_tier": best_tier,
        "reasons_expert": {
            "best_support": _compact_ref(support_best),
            "best_refute": _compact_ref(refute_best),
            "conflict": conflict,
            "penalties": penalties,
            "best_tier": best_tier,
            "penalty_mass": {
                "conflict_mass": conflict_mass,
                "temporal_mass": temporal_mass,
                "diversity_shortfall": diversity_shortfall,
            },
        },
    }


def _compact_ref(item: dict | None) -> dict | None:
    if not item:
        return None
    return {
        "tier": item.get("tier"),
        "url": item.get("url"),
        "quote": item.get("quote"),
    }
