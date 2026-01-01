# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence statistics builder for per-claim analysis.

Computes aggregate counts summarizing evidence coverage for a single claim.
"""

from __future__ import annotations

from spectrue_core.schema.claim_frame import EvidenceItemFrame, EvidenceStats


# High trust tiers (A, A', B are considered high trust)
HIGH_TRUST_TIERS = frozenset({"A", "A'", "B"})


def build_evidence_stats(evidence_items: tuple[EvidenceItemFrame, ...]) -> EvidenceStats:
    """
    Build aggregate evidence statistics from a list of evidence items.
    
    Args:
        evidence_items: Evidence items scoped to a single claim
    
    Returns:
        EvidenceStats with computed counts
    """
    if not evidence_items:
        return EvidenceStats(
            total_sources=0,
            support_sources=0,
            refute_sources=0,
            context_sources=0,
            high_trust_sources=0,
            direct_quotes=0,
            conflicting_evidence=False,
            missing_sources=True,
            missing_direct_quotes=True,
        )

    total = len(evidence_items)
    support_count = 0
    refute_count = 0
    context_count = 0
    high_trust_count = 0
    quote_count = 0

    for item in evidence_items:
        # Count by stance
        stance = (item.stance or "").upper()
        if stance == "SUPPORT":
            support_count += 1
        elif stance == "REFUTE":
            refute_count += 1
        elif stance == "CONTEXT":
            context_count += 1

        # Count high trust sources
        tier = (item.source_tier or "").upper()
        if tier in HIGH_TRUST_TIERS:
            high_trust_count += 1

        # Count direct quotes
        if item.quote and len(item.quote.strip()) > 10:
            quote_count += 1

    # Detect conflicting evidence (both support and refute present)
    conflicting = support_count > 0 and refute_count > 0

    return EvidenceStats(
        total_sources=total,
        support_sources=support_count,
        refute_sources=refute_count,
        context_sources=context_count,
        high_trust_sources=high_trust_count,
        direct_quotes=quote_count,
        conflicting_evidence=conflicting,
        missing_sources=(total == 0),
        missing_direct_quotes=(quote_count == 0),
    )


def compute_evidence_quality_score(stats: EvidenceStats) -> float:
    """
    Compute a simple evidence quality score from stats.
    
    Higher scores indicate better evidence coverage.
    
    Args:
        stats: Evidence statistics
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    if stats.total_sources == 0:
        return 0.0

    score = 0.0

    # Base score from source count (diminishing returns)
    source_score = min(1.0, stats.total_sources / 5.0) * 0.3
    score += source_score

    # Bonus for high trust sources
    if stats.high_trust_sources > 0:
        trust_score = min(1.0, stats.high_trust_sources / 3.0) * 0.25
        score += trust_score

    # Bonus for direct quotes
    if stats.direct_quotes > 0:
        quote_score = min(1.0, stats.direct_quotes / 2.0) * 0.25
        score += quote_score

    # Bonus for supporting evidence
    if stats.support_sources > 0 or stats.refute_sources > 0:
        stance_score = 0.2
        score += stance_score

    # Penalty for conflicting evidence (reduces confidence, not quality)
    # No penalty applied here as conflict can be informative

    return min(1.0, score)