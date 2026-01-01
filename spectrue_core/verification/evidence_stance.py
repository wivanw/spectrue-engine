# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence Stance Processing

Stance classification, conflict detection, and verdict state derivation.

M119: Extracted from pipeline_evidence.py as part of core logic modularization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


CANONICAL_VERDICT_STATES = {
    "supported",
    "refuted",
    "conflicted",
    "insufficient_evidence",
}


def count_stance_evidence(
    claim_id: str,
    evidence_items: list[dict[str, Any]],
) -> tuple[int, int, str | None]:
    """
    Count supporting and refuting evidence for a claim.
    
    Args:
        claim_id: Claim identifier
        evidence_items: List of evidence items from pack
        
    Returns:
        Tuple of (n_support, n_refute, best_tier)
    """
    n_support = 0
    n_refute = 0
    best_tier = None
    
    from spectrue_core.verification.evidence_explainability import get_tier_rank
    
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        item_claim_id = item.get("claim_id")
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        stance = str(item.get("stance") or "").lower()
        tier = item.get("tier")
        
        if stance in ("support", "sup", "supported"):
            n_support += 1
        elif stance in ("refute", "ref", "refuted"):
            n_refute += 1
        
        # Track best tier
        if tier and (
            best_tier is None or get_tier_rank(tier) > get_tier_rank(best_tier)
        ):
            best_tier = tier
    
    return n_support, n_refute, best_tier


def derive_verdict_state_from_llm_score(
    llm_score: float,
    n_support: int,
    n_refute: int,
) -> str:
    """
    Derive canonical verdict state from LLM score and evidence counts.
    
    Args:
        llm_score: LLM verdict score (0-1)
        n_support: Count of supporting evidence
        n_refute: Count of refuting evidence
        
    Returns:
        Canonical verdict state: "supported", "refuted", "conflicted", 
        or "insufficient_evidence"
    """
    if llm_score > 0.65:
        return "supported"
    elif llm_score < 0.35:
        return "refuted"
    elif n_support > 0 or n_refute > 0:
        return "conflicted"
    else:
        return "insufficient_evidence"


def derive_verdict_from_score(llm_score: float) -> str:
    """
    Derive verdict label from LLM score.
    
    Args:
        llm_score: LLM verdict score (0-1)
        
    Returns:
        Verdict label: "verified", "refuted", or "ambiguous"
    """
    if llm_score > 0.65:
        return "verified"
    elif llm_score < 0.35:
        return "refuted"
    else:
        return "ambiguous"


def detect_evidence_conflict(n_support: int, n_refute: int) -> bool:
    """
    Detect if evidence is conflicting.
    
    Args:
        n_support: Count of supporting evidence
        n_refute: Count of refuting evidence
        
    Returns:
        True if both support and refute evidence exist
    """
    return n_support > 0 and n_refute > 0


def check_has_direct_evidence(
    claim_id: str,
    evidence_items: list[dict[str, Any]],
) -> bool:
    """
    Check if claim has direct evidence (SUPPORT/REFUTE with quote).
    
    Args:
        claim_id: Claim identifier
        evidence_items: List of evidence items from pack
        
    Returns:
        True if direct evidence exists
    """
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        if claim_id and item.get("claim_id") not in (None, claim_id):
            continue
        
        stance = str(item.get("stance") or "").upper()
        if stance in ("SUPPORT", "REFUTE") and item.get("quote"):
            return True
    
    return False

