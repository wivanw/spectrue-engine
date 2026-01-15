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

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from spectrue_core.schema.claim_frame import (
        EvidenceStats as FrameEvidenceStats,
        ConfirmationCounts,
        EvidenceItemFrame
    )

@dataclass
class EvidenceStats:
    """
    Evidence-specific statistics, separated from BudgetState.
    
    Used to track quality signals like direct evidence, coverage slots,
    and distinct domains across the whole verification run.
    """
    sources_observed: int = 0
    sources_with_quote: int = 0
    direct_evidence: int = 0
    unique_domains: set[str] = field(default_factory=set)
    coverage_slots: set[str] = field(default_factory=set)

    def observe(self, ev: Any) -> None:
        """Observe an EvidenceItem or dict and update statistics."""
        self.sources_observed += 1

        # Handle both EvidenceItem objects and dicts
        quote = getattr(ev, "quote", None) if not isinstance(ev, dict) else ev.get("quote")
        if quote:
            self.sources_with_quote += 1

        role = getattr(ev, "evidence_role", "indirect") if not isinstance(ev, dict) else ev.get("evidence_role", "indirect")
        if role == "direct":
            self.direct_evidence += 1

        domain = getattr(ev, "domain", "") if not isinstance(ev, dict) else ev.get("domain", "")
        if domain:
            self.unique_domains.add(domain)

        covers = getattr(ev, "covers", []) if not isinstance(ev, dict) else ev.get("covers", [])
        for c in covers:
            self.coverage_slots.add(c)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a serializable dictionary."""
        return {
            "sources_observed": self.sources_observed,
            "sources_with_quote": self.sources_with_quote,
            "direct_evidence": self.direct_evidence,
            "unique_domain_count": len(self.unique_domains),
            "coverage_slots": list(self.coverage_slots),
        }


def build_evidence_stats(evidence_items: Tuple["EvidenceItemFrame", ...]) -> "FrameEvidenceStats":
    """
    Build EvidenceStats from a tuple of EvidenceItemFrames.
    """
    from spectrue_core.schema.claim_frame import (
        EvidenceStats as FrameEvidenceStats,
        EvidenceStanceStats,
    )

    total_sources = len(evidence_items)
    support_sources = 0
    refute_sources = 0
    context_sources = 0
    high_trust_sources = 0
    direct_quotes = 0
    unique_publishers = set()
    exact_dupes = 0
    similar_clusters = set()

    # Stance specific tracking
    support_publishers = set()
    refute_publishers = set()
    support_clusters = set()
    refute_clusters = set()

    seen_content_hashes = set()

    for item in evidence_items:
        # Deduplication check
        if item.content_hash:
            if item.content_hash in seen_content_hashes:
                exact_dupes += 1
            else:
                seen_content_hashes.add(item.content_hash)
        
        # Publisher tracking
        if item.publisher_id:
            unique_publishers.add(item.publisher_id)

        # Cluster tracking
        if item.similar_cluster_id:
            similar_clusters.add(item.similar_cluster_id)

        # Stance counting
        stance = (item.stance or "").upper()
        if stance in ("SUPPORT", "SUP"):
            support_sources += 1
            if item.publisher_id:
                support_publishers.add(item.publisher_id)
            if item.similar_cluster_id:
                support_clusters.add(item.similar_cluster_id)
        elif stance in ("REFUTE", "REF"):
            refute_sources += 1
            if item.publisher_id:
                refute_publishers.add(item.publisher_id)
            if item.similar_cluster_id:
                refute_clusters.add(item.similar_cluster_id)
        else:
            context_sources += 1

        # Quality signals
        if item.quote:
            direct_quotes += 1
        
        tier = (item.source_tier or "").upper()
        if tier in ("A", "A'", "A_PRIME"):
            high_trust_sources += 1

    return FrameEvidenceStats(
        total_sources=total_sources,
        support_sources=support_sources,
        refute_sources=refute_sources,
        context_sources=context_sources,
        high_trust_sources=high_trust_sources,
        direct_quotes=direct_quotes,
        conflicting_evidence=(support_sources > 0 and refute_sources > 0),
        missing_sources=(total_sources == 0),
        missing_direct_quotes=(direct_quotes == 0),
        exact_dupes_total=exact_dupes,
        similar_clusters_total=len(similar_clusters),
        publishers_total=len(unique_publishers),
        support=EvidenceStanceStats(
            precision_publishers=len(support_publishers),
            corroboration_clusters=len(support_clusters),
        ),
        refute=EvidenceStanceStats(
            precision_publishers=len(refute_publishers),
            corroboration_clusters=len(refute_clusters),
        ),
    )


def build_confirmation_counts(
    evidence_items: Tuple["EvidenceItemFrame", ...], 
    lambda_weight: float = 0.5
) -> "ConfirmationCounts":
    """
    Build ConfirmationCounts (C-scores) from evidence items.
    """
    from spectrue_core.schema.claim_frame import ConfirmationCounts

    c_precise = 0.0
    c_corr = 0.0
    
    # Simple heuristic accumulation based on stance and tier
    # Real implementation would use probabilistic mixing
    for item in evidence_items:
        stance = (item.stance or "").upper()
        if stance in ("SUPPORT", "REFUTE"):
            weight = 1.0
            tier = (item.source_tier or "").upper()
            if tier == "A":
                weight = 1.0
            elif tier == "B":
                weight = 0.7
            else:
                weight = 0.4
            
            # Attribute to precise or corr based on unknown logic, 
            # for now split evenly or based on attribution type?
            # Assuming attribution="precision" vs "corroboration"
            attr = item.attribution or "corroboration"
            if attr == "precision":
                c_precise += weight
            else:
                c_corr += weight

    c_total = (lambda_weight * c_precise) + ((1 - lambda_weight) * c_corr)

    return ConfirmationCounts(
        C_precise=c_precise,
        C_corr=c_corr,
        C_total=c_total
    )
