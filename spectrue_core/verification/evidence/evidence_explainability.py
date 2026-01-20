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


def compute_alpha_cap(
    *,
    independent_source_count: int,
    direct_anchor_count: int,
) -> float:
    """
    Compute Alpha cap based on source diversity and anchors.
    
    cap_independence = 1 - exp(-0.8 * independent_source_count)
    cap_anchors = 1 - exp(-1.0 * direct_anchor_count)
    A_cap = min(1.0, max(cap_anchors, 0.2) * max(cap_independence, 0.4))
    """
    cap_indep = 1.0 - math.exp(-0.8 * float(independent_source_count))
    cap_anchors = 1.0 - math.exp(-1.0 * float(direct_anchor_count))
    
    a_cap = min(1.0, max(cap_anchors, 0.2) * max(cap_indep, 0.4))
    return float(a_cap)


def compute_explainability_tier_adjustment(
    explainability_score: float,
    best_tier: str | None,
    claim_id: str,
) -> float | None:
    """
    DEPRECATED (M119): Logic moved to assign_claim_rgba and compute_alpha_cap.
    
    This function used to apply tier-based multiplicative adjustments. 
    Now it returns None to signal no change from this module.
    """
    return None


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

