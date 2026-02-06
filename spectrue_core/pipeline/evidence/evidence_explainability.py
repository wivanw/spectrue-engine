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
from typing import Any

logger = logging.getLogger(__name__)


def norm_claim_id(x: Any) -> str | None:
    """Normalize claim ID for consistent lookup."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null", "undefined"):
        return None
    return s


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


# compute_alpha_cap removed in M133 — LLM A-score passes through unchanged
# Rationale: caps were forcing A ≤ 0.4 even with good evidence; now trust LLM judgment



# compute_explainability_tier_adjustment removed in M133 — was already deprecated (M119)


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

