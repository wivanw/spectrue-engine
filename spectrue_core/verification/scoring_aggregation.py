# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Deterministic claim verdict aggregation.

Implements tier-dominant, conflict-aware scoring with penalties
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
    "See M104 FR-010.",
    DeprecationWarning,
    stacklevel=2,
)

TIER_RANK = {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}
TIER_CEILING = {"D": 0.35, "C": 0.55, "B": 0.75, "A'": 0.75, "A": 0.90}
REFUTE_SCORE = {"D": 0.45, "C": 0.35, "B": 0.25, "A'": 0.20, "A": 0.10}

AMBIGUOUS_MIN = 0.35
AMBIGUOUS_MAX = 0.65


def _tier_rank(tier: str | None) -> int:
    if not tier:
        return 0
    return TIER_RANK.get(str(tier).strip().upper(), 0)


def _tier_ceiling(tier: str | None) -> float:
    if not tier:
        return 0.5
    return TIER_CEILING.get(str(tier).strip().upper(), 0.5)


def _score_from_tier(tier: str | None, *, refute: bool) -> float:
    if not tier:
        return 0.5
    t = str(tier).strip().upper()
    if refute:
        return REFUTE_SCORE.get(t, 0.3)
    return _tier_ceiling(t)


def _is_comparable_tier(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return abs(_tier_rank(a) - _tier_rank(b)) <= 1


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _best_evidence(items: list[dict], *, stance: str) -> dict | None:
    best: dict | None = None
    best_rank = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("stance") or "").upper() != stance:
            continue
        if not item.get("quote"):
            continue
        tier = str(item.get("tier") or "").upper()
        rank = _tier_rank(tier)
        if rank > best_rank:
            best_rank = rank
            best = item
    return best


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

    support_tier = support_best.get("tier") if support_best else None
    refute_tier = refute_best.get("tier") if refute_best else None

    conflict = bool(support_best and refute_best and _is_comparable_tier(support_tier, refute_tier))

    if refute_best and (_tier_rank(refute_tier) >= _tier_rank(support_tier) or not support_best):
        base = _score_from_tier(refute_tier, refute=True)
        verdict = "refuted"
    else:
        base = _score_from_tier(support_tier, refute=False)
        verdict = "verified"

    penalties: list[str] = []
    if conflict:
        base -= 0.10
        penalties.append("conflict_penalty")

    is_time_sensitive = bool((temporality or {}).get("is_time_sensitive"))
    forced_ambiguous = False
    if is_time_sensitive:
        outdated = any(str(i.get("temporal_flag") or "").lower() == "outdated" for i in items)
        if outdated:
            base -= 0.15
            penalties.append("temporal_penalty")
            if verdict == "verified":
                verdict = "ambiguous"
                forced_ambiguous = True

    stats = evidence_pack.get("stats") or {}
    min_domain_div = int(policy.get("min_domain_diversity", 1))
    if isinstance(stats, dict) and stats.get("domain_diversity") is not None:
        if int(stats.get("domain_diversity") or 0) < min_domain_div:
            base -= 0.05
            penalties.append("consistency_penalty")

    best_tier = None
    if support_tier and refute_tier:
        best_tier = support_tier if _tier_rank(support_tier) >= _tier_rank(refute_tier) else refute_tier
    elif support_tier:
        best_tier = support_tier
    elif refute_tier:
        best_tier = refute_tier

    pre_ceiling = base
    ceiling = _tier_ceiling(best_tier)
    base = _clamp(base, 0.0, ceiling)

    if conflict:
        support_strength = _tier_rank(support_tier)
        refute_strength = _tier_rank(refute_tier)
        if abs(support_strength - refute_strength) <= 0:
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
            "ceiling": ceiling,
            "final_score": base,
            "conflict": conflict,
            "forced_ambiguous": forced_ambiguous,
        },
    )

    return {
        "verdict": verdict,
        "verdict_score": base,
        "reasons_expert": {
            "best_support": _compact_ref(support_best),
            "best_refute": _compact_ref(refute_best),
            "conflict": conflict,
            "penalties": penalties,
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
