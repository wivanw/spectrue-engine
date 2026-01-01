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
Weighted RGBA Aggregation

Fixes the "mixed article dilution" problem where non-verifiable claims
(horoscopes, predictions, opinions) were averaging down factual claim scores.

Solution:
- Each claim has a role_weight based on metadata
- verification_target=none → weight=0 (excluded from aggregate)
- Final RGBA is weighted average of verifiable claims only

Example:
    Article with 3 claims:
    1. "Biden won 2024 election" → verified=0.95, weight=1.0 (CORE/REALITY)
    2. "Economy will improve" → weight=0.0 (CONTEXT/NONE - prediction)
    3. "Source says X" → verified=0.80, weight=0.7 (ATTRIBUTION)
    
    Old: (0.95 + 0.50 + 0.80) / 3 = 0.75 (diluted)
    New: (0.95*1.0 + 0.50*0.0 + 0.80*0.7) / (1.0 + 0.7) = 0.53 + 0.33 = 0.89 (correct)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.schema.claim_metadata import ClaimMetadata

logger = logging.getLogger(__name__)


@dataclass
class ClaimScore:
    """Score for a single claim with weighting metadata."""
    claim_id: str
    verified_score: float  # 0-1
    danger_score: float    # 0-1
    style_score: float     # 0-1
    explainability_score: float  # 0-1

    # Weighting factors
    role_weight: float = 1.0  # From ClaimMetadata.role_weight
    check_worthiness: float = 0.5  # From ClaimMetadata
    evidence_quality: float = 1.0  # From sufficiency check

    @property
    def total_weight(self) -> float:
        """Calculate total weight for this claim."""
        return self.role_weight * self.check_worthiness * self.evidence_quality

    @property
    def is_excluded(self) -> bool:
        """Check if claim should be excluded from aggregate (weight=0)."""
        return self.role_weight == 0.0 or self.total_weight < 0.01


@dataclass
class AggregatedRGBA:
    """Aggregated RGBA scores from all claims."""
    verified: float = 0.5
    danger: float = 0.5
    style: float = 0.5
    explainability: float = 0.5

    # Metadata
    total_claims: int = 0
    included_claims: int = 0  # Claims with weight > 0
    excluded_claims: int = 0  # Claims with weight = 0
    total_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "verified": round(self.verified, 3),
            "danger": round(self.danger, 3),
            "style": round(self.style, 3),
            "explainability": round(self.explainability, 3),
            "meta": {
                "total_claims": self.total_claims,
                "included_claims": self.included_claims,
                "excluded_claims": self.excluded_claims,
                "total_weight": round(self.total_weight, 3),
            },
        }


def aggregate_weighted(claim_scores: list[ClaimScore]) -> AggregatedRGBA:
    """
    T30: Aggregate RGBA scores with role-based weighting.
    
    Weight formula:
        weight = role_weight × check_worthiness × evidence_quality
    
    Where:
        - role_weight: 0.0 for CONTEXT/META, 1.0 for CORE, etc.
        - check_worthiness: LLM's priority score (0-1)
        - evidence_quality: Based on sufficiency check (default 1.0)
    
    Aggregation:
        score = Σ(claim_score × weight) / Σ(weight)
    
    Args:
        claim_scores: List of ClaimScore objects
        
    Returns:
        AggregatedRGBA with weighted scores
    """
    result = AggregatedRGBA()

    if not claim_scores:
        return result

    result.total_claims = len(claim_scores)

    # Accumulators
    weighted_verified = 0.0
    weighted_danger = 0.0
    weighted_style = 0.0
    weighted_explainability = 0.0
    total_weight = 0.0

    for cs in claim_scores:
        weight = cs.total_weight

        if cs.is_excluded:
            result.excluded_claims += 1
            continue

        result.included_claims += 1
        total_weight += weight

        weighted_verified += cs.verified_score * weight
        weighted_danger += cs.danger_score * weight
        weighted_style += cs.style_score * weight
        weighted_explainability += cs.explainability_score * weight

    result.total_weight = total_weight

    # Calculate weighted averages
    if total_weight > 0:
        result.verified = weighted_verified / total_weight
        result.danger = weighted_danger / total_weight
        result.style = weighted_style / total_weight
        result.explainability = weighted_explainability / total_weight
    else:
        # No verifiable claims - return neutral scores
        result.verified = 0.5
        result.danger = 0.5
        result.style = 0.5
        result.explainability = 0.5
        logger.warning(
            "[M80] No verifiable claims (all weights=0). "
            "Returning neutral RGBA scores."
        )

    return result


def claim_to_score(
    claim: dict,
    *,
    verified_score: float,
    danger_score: float,
    style_score: float = 0.5,
    explainability_score: float = 0.5,
    evidence_quality: float = 1.0,
) -> ClaimScore:
    """
    T31: Convert claim dict to ClaimScore using metadata.
    
    Args:
        claim: Claim dict with metadata
        verified_score: 0-1 verification score from LLM
        danger_score: 0-1 danger score from LLM
        style_score: 0-1 style score
        explainability_score: 0-1 explainability score
        evidence_quality: 0-1 evidence quality factor
        
    Returns:
        ClaimScore with proper weighting
    """
    claim_id = claim.get("id", "unknown")

    # Get metadata
    metadata = claim.get("metadata")
    if metadata and isinstance(metadata, ClaimMetadata):
        role_weight = metadata.role_weight
        check_worthiness = metadata.check_worthiness
    else:
        # Default: full weight for backward compat
        role_weight = 1.0
        check_worthiness = 0.5

    return ClaimScore(
        claim_id=claim_id,
        verified_score=verified_score,
        danger_score=danger_score,
        style_score=style_score,
        explainability_score=explainability_score,
        role_weight=role_weight,
        check_worthiness=check_worthiness,
        evidence_quality=evidence_quality,
    )


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def recompute_verified_score(claim_verdicts: list[dict]) -> float | None:
    """Recompute overall verified_score from claim verdicts."""
    scores = []
    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        score = _safe_score(cv.get("verdict_score"))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def apply_conflict_explainability_penalty(
    explainability_score: float,
    *,
    penalty: float = 0.15,
) -> float:
    """Reduce explainability when strong evidence conflicts are detected."""
    return max(0.0, explainability_score - penalty)


def apply_dependency_penalties(
    claim_verdicts: list[dict],
    claims: list[dict],
    *,
    refute_threshold: float = 0.2,
    cap_on_refute: float = 0.4,
) -> bool:
    """
    Propagate premise failures to dependent conclusions.

    If any dependency is refuted (score <= refute_threshold),
    cap the dependent claim's verdict_score to cap_on_refute.
    """
    if not claim_verdicts or not claims:
        return False

    verdict_by_id: dict[str, dict] = {}
    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        cid = cv.get("claim_id")
        if cid:
            verdict_by_id[str(cid)] = cv

    changed = False

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = claim.get("id")
        if not claim_id:
            continue
        structure = claim.get("structure")
        if not isinstance(structure, dict):
            continue
        deps = structure.get("dependencies", [])
        if not isinstance(deps, list) or not deps:
            continue

        refuted = False
        for dep_id in deps:
            dep_verdict = verdict_by_id.get(dep_id)
            if not dep_verdict:
                continue
            dep_score = _safe_score(dep_verdict.get("verdict_score"))
            if dep_score is not None and dep_score <= refute_threshold:
                refuted = True
                break

        if not refuted:
            continue

        verdict = verdict_by_id.get(str(claim_id))
        if not verdict:
            continue
        current_score = _safe_score(verdict.get("verdict_score"))
        if current_score is None:
            continue

        if current_score > cap_on_refute:
            verdict["verdict_score"] = cap_on_refute
            verdict["dependency_penalty"] = "premise_refuted"
            changed = True

    return changed
