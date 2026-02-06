# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Weighted RGBA Aggregation logic."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.domain.claims.model import ClaimMetadata

logger = logging.getLogger(__name__)


@dataclass
class ClaimScore:
    """Score for a single claim with weighting metadata."""
    claim_id: str
    verified_score: float | None  # 0-1
    danger_score: float | None    # 0-1
    style_score: float | None     # 0-1
    explainability_score: float | None  # 0-1

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
    verified: float | None = None
    danger: float | None = None
    style: float | None = None
    explainability: float | None = None

    # Metadata
    total_claims: int = 0
    included_claims: int = 0  # Claims with weight > 0
    excluded_claims: int = 0  # Claims with weight = 0
    total_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "verified": round(self.verified, 3) if self.verified is not None else None,
            "danger": round(self.danger, 3) if self.danger is not None else None,
            "style": round(self.style, 3) if self.style is not None else None,
            "explainability": round(self.explainability, 3) if self.explainability is not None else None,
            "meta": {
                "total_claims": self.total_claims,
                "included_claims": self.included_claims,
                "excluded_claims": self.excluded_claims,
                "total_weight": round(self.total_weight, 3),
            },
        }


def aggregate_weighted(claim_scores: list[ClaimScore]) -> AggregatedRGBA:
    """
    Aggregate RGBA scores with role-based weighting.
    """
    result = AggregatedRGBA()

    if not claim_scores:
        return result

    result.total_claims = len(claim_scores)

    # Accumulators
    weighted_verified = 0.0
    weight_v = 0.0
    weighted_danger = 0.0
    weight_r = 0.0
    weighted_style = 0.0
    weight_b = 0.0
    weighted_explainability = 0.0
    weight_a = 0.0

    for cs in claim_scores:
        weight = cs.total_weight

        if cs.is_excluded:
            result.excluded_claims += 1
            continue

        result.included_claims += 1

        if cs.verified_score is not None:
            weighted_verified += cs.verified_score * weight
            weight_v += weight
        if cs.danger_score is not None:
            weighted_danger += cs.danger_score * weight
            weight_r += weight
        if cs.style_score is not None:
            weighted_style += cs.style_score * weight
            weight_b += weight
        if cs.explainability_score is not None:
            weighted_explainability += cs.explainability_score * weight
            weight_a += weight

    result.total_weight = weight_v

    # Calculate weighted averages
    if weight_v > 0:
        result.verified = weighted_verified / weight_v
    if weight_r > 0:
        result.danger = weighted_danger / weight_r
    if weight_b > 0:
        result.style = weighted_style / weight_b
    if weight_a > 0:
        result.explainability = weighted_explainability / weight_a
    
    return result


def claim_to_score(
    claim: dict,
    *,
    verified_score: float | None,
    danger_score: float | None,
    style_score: float | None = None,
    explainability_score: float | None = None,
    evidence_quality: float = 1.0,
) -> ClaimScore:
    """
    Convert claim dict to ClaimScore using metadata.
    """
    claim_id = claim.get("id", "unknown")

    # Get metadata
    metadata = claim.get("metadata")
    if metadata and isinstance(metadata, ClaimMetadata):
        from spectrue_core.domain.claims.policy import get_role_weight
        role_weight = get_role_weight(metadata)
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
