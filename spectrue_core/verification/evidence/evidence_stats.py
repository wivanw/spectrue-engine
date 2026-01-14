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

from spectrue_core.schema.claim_frame import (
    ConfirmationCounts,
    EvidenceItemFrame,
    EvidenceStats,
    EvidenceStanceStats,
)


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
            exact_dupes_total=0,
            similar_clusters_total=0,
            publishers_total=0,
            support=EvidenceStanceStats(),
            refute=EvidenceStanceStats(),
        )

    total = len(evidence_items)
    support_count = 0
    refute_count = 0
    context_count = 0
    high_trust_count = 0
    quote_count = 0

    hash_counts: dict[str, int] = {}
    publisher_ids: set[str] = set()
    similar_clusters: set[str] = set()
    support_precise_publishers: set[str] = set()
    support_corr_clusters: set[str] = set()
    refute_precise_publishers: set[str] = set()
    refute_corr_clusters: set[str] = set()

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

        # Duplicate + similarity stats
        content_hash = (item.content_hash or "").strip()
        if content_hash:
            hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1

        publisher_id = (item.publisher_id or "").strip()
        if publisher_id:
            publisher_ids.add(publisher_id)

        sim_cluster_id = (item.similar_cluster_id or "").strip()
        if sim_cluster_id:
            similar_clusters.add(sim_cluster_id)

        attribution = (item.attribution or "").lower()
        if stance == "SUPPORT":
            if attribution == "precision" and publisher_id:
                support_precise_publishers.add(publisher_id)
            elif attribution == "corroboration" and sim_cluster_id:
                support_corr_clusters.add(sim_cluster_id)
        elif stance == "REFUTE":
            if attribution == "precision" and publisher_id:
                refute_precise_publishers.add(publisher_id)
            elif attribution == "corroboration" and sim_cluster_id:
                refute_corr_clusters.add(sim_cluster_id)

    # Detect conflicting evidence (both support and refute present)
    conflicting = support_count > 0 and refute_count > 0

    exact_dupes_total = sum(count for count in hash_counts.values() if count > 1)

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
        exact_dupes_total=exact_dupes_total,
        similar_clusters_total=len(similar_clusters),
        publishers_total=len(publisher_ids),
        support=EvidenceStanceStats(
            precision_publishers=len(support_precise_publishers),
            corroboration_clusters=len(support_corr_clusters),
        ),
        refute=EvidenceStanceStats(
            precision_publishers=len(refute_precise_publishers),
            corroboration_clusters=len(refute_corr_clusters),
        ),
    )


def build_confirmation_counts(
    evidence_items: tuple[EvidenceItemFrame, ...],
    *,
    lambda_weight: float,
) -> ConfirmationCounts:
    support_publishers: set[str] = set()
    support_clusters: set[str] = set()

    for item in evidence_items:
        if (item.stance or "").upper() != "SUPPORT":
            continue
        attribution = (item.attribution or "").lower()
        if attribution == "precision":
            publisher_id = (item.publisher_id or "").strip()
            if publisher_id:
                support_publishers.add(publisher_id)
        elif attribution == "corroboration":
            cluster_id = (item.similar_cluster_id or "").strip()
            if cluster_id:
                support_clusters.add(cluster_id)

    c_precise = float(len(support_publishers))
    c_corr = float(len(support_clusters))
    c_total = c_precise + float(lambda_weight) * c_corr
    return ConfirmationCounts(C_precise=c_precise, C_corr=c_corr, C_total=c_total)


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
