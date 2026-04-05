# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Domain logic for stance evaluation and conflict detection."""

from __future__ import annotations
from typing import Any


def detect_evidence_conflict(n_support: int, n_refute: int) -> bool:
    """
    Detect if evidence is conflicting.
    
    Returns True if both support and refute evidence exist.
    """
    return n_support > 0 and n_refute > 0


def check_has_direct_evidence(
    claim_id: str | None,
    evidence_items: list[Any],
) -> bool:
    """
    Check if claim has direct evidence (SUPPORT/REFUTE with quote).
    """
    for item in evidence_items:
        # Support both domain models and legacy dicts
        if hasattr(item, "claim_id"):
            item_claim_id = item.claim_id
            stance = str(getattr(item, "stance", "") or "").upper()
            has_quote = bool(getattr(item, "quote", ""))
        else:
            item_claim_id = item.get("claim_id")
            stance = str(item.get("stance") or "").upper()
            has_quote = bool(item.get("quote"))

        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        if stance in ("SUPPORT", "REFUTE") and has_quote:
            return True
    
    return False


def count_stance_evidence(
    claim_id: str | None,
    evidence_items: list[Any],
) -> tuple[int, int, str | None]:
    """
    Count supporting and refuting evidence for a claim and determine best tier.
    """
    n_support = 0
    n_refute = 0
    best_tier = None
    
    # Tier ranking for comparison
    TIER_RANK = {"D": 1, "C": 2, "B": 3, "A'": 4, "A": 4}
    
    for item in evidence_items:
        if hasattr(item, "claim_id"):
            item_claim_id = item.claim_id
            stance = str(getattr(item, "stance", "") or "").lower()
            tier = getattr(item, "tier", None)
        else:
            item_claim_id = item.get("claim_id")
            stance = str(item.get("stance") or "").lower()
            tier = item.get("tier")
            
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        match stance:
            case "support" | "sup" | "supported":
                n_support += 1
            case "refute" | "ref" | "refuted":
                n_refute += 1
        
        # Track best tier
        if tier:
            rank = TIER_RANK.get(str(tier).upper(), 0)
            best_rank = TIER_RANK.get(str(best_tier).upper(), 0) if best_tier else -1
            if rank > best_rank:
                best_tier = tier
    
    return n_support, n_refute, best_tier
