# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Evidence Sufficiency Check (Use Case)

Determines if collected evidence is sufficient to make a verdict,
enabling early exit from progressive widening.
"""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.domain.claims.model import (
    VerificationTarget,
)
from spectrue_core.domain.claims.sufficiency import (
    SufficiencyStatus,
    SufficiencyDecision,
    SufficiencyResult,
    SufficiencyDecisionResult,
    evidence_sufficiency,
    _extract_domain,
)
from spectrue_core.domain.verification.search.search_policy import SearchPolicyProfile

logger = logging.getLogger(__name__)


def check_sufficiency_for_claim(
    claim: dict,
    sources: list[Any],
) -> SufficiencyResult:
    """
    Convenience wrapper that extracts metadata from claim.
    
    Args:
        claim: Claim dict with id, metadata, normalized_text
        sources: List of sources for this claim
        
    Returns:
        SufficiencyResult
    """
    claim_id = claim.get("id", "unknown")
    claim_text = claim.get("normalized_text", "") or claim.get("text", "")

    metadata = claim.get("metadata")
    if metadata:
        verification_target = metadata.verification_target
        use_policy_by_channel = getattr(metadata.retrieval_policy, "use_policy_by_channel", {}) if getattr(metadata, "retrieval_policy", None) else {}
    else:
        verification_target = VerificationTarget.REALITY
        use_policy_by_channel = {}

    return evidence_sufficiency(
        claim_id=claim_id,
        sources=sources,
        verification_target=verification_target,
        claim_text=claim_text,
        use_policy_by_channel=use_policy_by_channel,
    )


def _estimate_coverage(sources: list[Any], min_relevance_score: float) -> float:
    if not sources:
        return 0.0
    hits = 0
    for src in sources:
        if not isinstance(src, dict):
            continue
        score = src.get("relevance_score", 0.0)
        try:
            if float(score or 0.0) >= float(min_relevance_score):
                hits += 1
        except Exception:
            continue
    return hits / len(sources)


def _estimate_diversity(sources: list[Any]) -> float:
    if not sources:
        return 0.0
    domains = set()
    for src in sources:
        if not isinstance(src, dict):
            continue
        url = src.get("url", "") or src.get("link", "")
        domain = _extract_domain(str(url))
        if domain:
            domains.add(domain)
    return len(domains) / len(sources)


def judge_sufficiency_for_claim(
    claim: dict,
    sources: list[Any],
    *,
    policy_profile: SearchPolicyProfile | None = None,
    use_policy_by_channel: dict[str, Any] | None = None,
) -> SufficiencyDecisionResult:
    """
    Decide whether to stop or continue iterative retrieval.

    Uses sufficiency rules, then applies policy thresholds to determine
    ENOUGH vs NEED_FOLLOWUP. Falls back to STOP on errors.
    """
    claim_id = claim.get("id", "unknown") if isinstance(claim, dict) else "unknown"

    try:
        metadata = claim.get("metadata") if isinstance(claim, dict) else getattr(claim, "metadata", None)
        if use_policy_by_channel is None:
            use_policy_by_channel = (
                getattr(metadata.retrieval_policy, "use_policy_by_channel", {})
                if getattr(metadata, "retrieval_policy", None)
                else {}
            )

        verification_target = VerificationTarget.REALITY
        if metadata and getattr(metadata, "verification_target", None):
            verification_target = metadata.verification_target

        sufficiency = evidence_sufficiency(
            claim_id=claim_id,
            sources=sources,
            verification_target=verification_target,
            claim_text=claim.get("normalized_text", "") or claim.get("text", ""),
            use_policy_by_channel=use_policy_by_channel,
        )

        degraded = "lead-only" in sufficiency.reason.lower() or "lack usable evidence" in sufficiency.reason.lower()

        if sufficiency.status == SufficiencyStatus.SKIP:
            decision = SufficiencyDecision.STOP
        elif sufficiency.status == SufficiencyStatus.SUFFICIENT:
            decision = SufficiencyDecision.ENOUGH
        else:
            decision = SufficiencyDecision.NEED_FOLLOWUP

        coverage = 0.0
        diversity = 0.0
        aligned_hits = 0
        if policy_profile is not None and decision != SufficiencyDecision.STOP:
            thresholds = policy_profile.quality_thresholds
            coverage = _estimate_coverage(sources, thresholds.min_relevance_score)
            diversity = _estimate_diversity(sources)
            aligned_hits = sum(
                1
                for src in sources
                if isinstance(src, dict)
                and float(src.get("relevance_score") or 0.0) >= float(thresholds.min_relevance_score)
                and str(src.get("claim_id") or claim_id) == str(claim_id)
            )
            if (
                coverage < thresholds.min_coverage
                or diversity < thresholds.min_diversity
                or aligned_hits == 0
            ):
                decision = SufficiencyDecision.NEED_FOLLOWUP
                sufficiency.reason = (
                    f"{sufficiency.reason}; below quality thresholds "
                    f"(coverage={coverage:.2f}, diversity={diversity:.2f}, aligned_hits={aligned_hits})"
                )

            if not policy_profile.stop_conditions.stop_on_sufficiency and decision == SufficiencyDecision.ENOUGH:
                decision = SufficiencyDecision.NEED_FOLLOWUP
                sufficiency.reason = "Policy requires follow-up despite sufficiency"

        return SufficiencyDecisionResult(
            claim_id=claim_id,
            decision=decision,
            reason=sufficiency.reason,
            rule_matched=sufficiency.rule_matched,
            degraded_confidence=degraded,
            coverage=coverage,
            diversity=diversity,
        )
    except Exception as exc:
        logger.warning("[M92] Sufficiency judge failed for %s: %s", claim_id, exc)
        return SufficiencyDecisionResult(
            claim_id=claim_id,
            decision=SufficiencyDecision.STOP,
            reason="Sufficiency judge failed",
            degraded_confidence=True,
        )