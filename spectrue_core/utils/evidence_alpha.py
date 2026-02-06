# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence Alpha (Deterministic Explainability)

Implements deterministic A_det via log-odds accumulation of source reliability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.verification.evidence.evidence_pack import EvidenceItem


def sigmoid(x: float) -> float:
    """Compute sigmoid of x."""
    try:
        if x > 100:
            return 1.0
        if x < -100:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 1.0 if x > 0 else 0.0


def weight_func(r: float) -> float:
    """
    Compute log-odds weight for reliability r.
    w(r) = ln(r/(1-r)) for r in (0.5, 0.999)
    if r <= 0.5 => 0
    """
    if r <= 0.5:
        return 0.0
    
    # Cap r to avoid log(0)
    r_capped = min(max(r, 0.50001), 0.999)
    return math.log(r_capped / (1.0 - r_capped))


def compute_A_det(items: list[EvidenceItem], claim_id: str | None) -> float:
    """
    Compute deterministic explainability score A_det.
    
    A_det = sigmoid(sum(coverage * w(r_eff)))
    
    Considers only items where:
    - item.claim_id matches or is None
    - item.stance == "SUPPORT"
    - item.quote exists (direct anchor)
    - item.r_eff is not None
    """
    from spectrue_core.verification.evidence.evidence_explainability import norm_claim_id
    target_cid = norm_claim_id(claim_id)
    sum_contrib = 0.0
    
    for item in items:
        # 1. Claim ID filter
        item_cid = norm_claim_id(item.get("claim_id"))
        if target_cid and item_cid not in (None, target_cid):
            continue
            
        # 2. Anchor filter: MUST have quote and NOT be IRRELEVANT
        stance = (item.get("stance") or "").upper()
        if stance == "IRRELEVANT":
            continue
            
        if not item.get("quote"):
            continue
            
        # 3. Reliability filter
        r_eff = item.get("r_eff")
        if r_eff is None:
            continue
            
        # Compute contribution
        # coverage = item.get("coverage", 1.0) # Coverage not yet in EvidenceItem, default to 1.0
        coverage = 1.0
        
        contrib = coverage * weight_func(r_eff)
        sum_contrib += contrib
        
    return sigmoid(sum_contrib)
