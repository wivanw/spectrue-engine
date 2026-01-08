# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence Explainability

Tier-based explainability factor computation for RGBA Alpha channel.

Extracted from pipeline_evidence.py as part of core logic modularization.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from spectrue_core.verification.evidence.evidence_scoring import (
    logit,
    sigmoid,
    TIER_A_BASELINE,
)
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def get_tier_rank(tier: str | None) -> int:
    """
    Get numeric rank for evidence tier.
    
    Args:
        tier: Tier label (A, A', B, C, D, or None)
        
    Returns:
        Numeric rank (0-4), higher is better
    """
    if not tier:
        return 0
    return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(
        str(tier).strip().upper(), 0
    )


def compute_explainability_tier_adjustment(
    explainability_score: float,
    best_tier: str | None,
    claim_id: str,
) -> float | None:
    """
    Apply tier-based adjustment to explainability score.
    
    Uses logit-space adjustment to preserve probabilistic properties.
    
    Args:
        explainability_score: Raw explainability from LLM (0-1)
        best_tier: Best evidence tier for this claim
        claim_id: Claim identifier for tracing
        
    Returns:
        Adjusted explainability score, or None if no adjustment needed
    """
    from spectrue_core.verification.evidence.evidence_scoring import explainability_factor_for_tier
    
    if not isinstance(explainability_score, (int, float)) or explainability_score < 0:
        return None
    
    pre_a = float(explainability_score)
    
    # Contract: pre_A must be strictly in (0, 1) for logit
    if not (0.0 < pre_a < 1.0) or not math.isfinite(pre_a):
        Trace.event(
            "verdict.explainability_missing",
            {"claim_id": claim_id, "best_tier": best_tier, "pre_A": pre_a},
        )
        return None
    
    factor, source, prior = explainability_factor_for_tier(best_tier)
    
    if factor <= 0 or not math.isfinite(factor):
        Trace.event(
            "verdict.explainability_bad_factor",
            {
                "claim_id": claim_id,
                "best_tier": best_tier,
                "pre_A": pre_a,
                "prior": prior,
                "baseline": TIER_A_BASELINE,
                "factor": factor,
                "source": source,
            },
        )
        return None
    
    # Apply logit-space adjustment
    post_a = sigmoid(logit(pre_a) + math.log(factor))
    
    # Only return if adjustment is significant
    if abs(post_a - pre_a) < 1e-9:
        return None
    
    Trace.event(
        "verdict.explainability_tier_factor",
        {
            "claim_id": claim_id,
            "best_tier": best_tier,
            "pre_A": pre_a,
            "prior": prior,
            "baseline": TIER_A_BASELINE,
            "factor": factor,
            "post_A": post_a,
            "source": source,
        },
    )
    
    return post_a


def find_best_tier_for_claim(
    claim_id: str,
    evidence_items: list[dict[str, Any]],
) -> str | None:
    """
    Find best (highest-ranked) evidence tier for a claim.
    
    Args:
        claim_id: Claim identifier
        evidence_items: List of evidence items from pack
        
    Returns:
        Best tier label (A, A', B, C, D) or None
    """
    best_tier = None
    
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        item_claim_id = item.get("claim_id")
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        tier = item.get("tier")
        if tier and (
            best_tier is None or get_tier_rank(tier) > get_tier_rank(best_tier)
        ):
            best_tier = tier
    
    return best_tier

