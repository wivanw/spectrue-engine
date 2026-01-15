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

Extracted from pipeline_evidence.py as part of core logic modularization.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.pipeline.mode import ScoringMode

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
    
    from spectrue_core.verification.evidence.evidence_explainability import get_tier_rank
    
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        item_claim_id = item.get("claim_id")
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        stance = str(item.get("stance") or "").lower()
        tier = item.get("tier")
        
        match stance:
            case "support" | "sup" | "supported":
                n_support += 1
            case "refute" | "ref" | "refuted":
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


def assign_claim_rgba(
    claim_verdict: dict[str, Any],
    *,
    global_r: float,
    global_b: float,
    global_a: float,
    judge_mode: ScoringMode | str,
) -> None:
    """
    Assign RGBA to claim verdict based on mode.
    
    Deep mode: Preserve existing RGBA from LLM judge (1:1 contract)
    Standard mode: Compute from global values
    
    Args:
        claim_verdict: Claim verdict dict (mutated in-place)
        global_r: Global danger score
        global_b: Global bias score
        global_a: Global explainability score
        judge_mode: ScoringMode enum or string
    """
    from spectrue_core.utils.trace import Trace
    
    cid = claim_verdict.get("claim_id", "unknown")
    g_score = float(claim_verdict.get("verdict_score", 0.5) or 0.5)
    existing_rgba = claim_verdict.get("rgba")
    
    # Handle enum or string input
    from spectrue_core.pipeline.mode import ScoringMode
    mode_value = judge_mode.value if isinstance(judge_mode, ScoringMode) else judge_mode
    
    has_valid_rgba = (
        isinstance(existing_rgba, list)
        and len(existing_rgba) == 4
        and all(isinstance(x, (int, float)) for x in existing_rgba)
    )
    
    # Determine source and final RGBA in one pass
    if has_valid_rgba:
        # Deep claim-judge already provided RGBA - keep LLM output 1:1
        Trace.event(
            "cv.rgba_assigned",
            {
                "claim_id": cid,
                "judge_mode": mode_value,
                "source": "llm",
                "rgba": existing_rgba,
            },
        )
    elif mode_value == ScoringMode.DEEP.value:
        # Deep mode but no RGBA from LLM - error handling
        if claim_verdict.get("status") == "error":
            # Expected: error claim has no RGBA
            Trace.event(
                "cv.rgba_assigned",
                {
                    "claim_id": cid,
                    "judge_mode": mode_value,
                    "source": "error",
                    "rgba": None,
                    "error_type": claim_verdict.get("error_type"),
                },
            )
        else:
            # Unexpected: valid claim missing RGBA
            claim_verdict["rgba"] = None
            claim_verdict["rgba_error"] = "missing_from_llm"
            Trace.event(
                "cv.rgba_assigned",
                {
                    "claim_id": cid,
                    "judge_mode": mode_value,
                    "source": "missing",
                    "rgba": None,
                    "error": "Deep mode claim missing RGBA from judge",
                },
            )
            logger.warning(
                "[DeepMode] Claim %s missing RGBA from judge - marked as error",
                cid,
            )
    else:
        # Standard mode fallback: compute from global values
        # Support per-claim explainability adjustment (Tier masking) if present
        local_a = claim_verdict.get("local_explainability")
        final_a = float(local_a) if local_a is not None and float(local_a) >= 0 else global_a
        
        claim_verdict["rgba"] = [global_r, g_score, global_b, final_a]
        
        # Log tier-factor info only when it's actually applied
        trace_data = {
            "claim_id": cid,
            "judge_mode": mode_value,
            "source": "fallback",
            "rgba": claim_verdict["rgba"],
        }
        if local_a is not None:
            trace_data["tier_factor_applied"] = True
            trace_data["global_a"] = global_a
            trace_data["local_a"] = local_a
        
        Trace.event("cv.rgba_assigned", trace_data)


def enrich_claim_sources(
    claim_verdict: dict[str, Any],
    all_sources: list[dict[str, Any]],
    enrich_func: Any,
) -> None:
    """
    Filter and enrich sources for a specific claim.
    
    Args:
        claim_verdict: Claim verdict dict (mutated in-place)
        all_sources: All available sources (scored + context)
        enrich_func: Function to enrich sources with trust metadata
    """
    from spectrue_core.verification.evidence.evidence_scoring import norm_id
    
    cid = norm_id(claim_verdict.get("claim_id"))
    if not cid:
        claim_verdict["sources"] = []
        return
    
    # Filter sources for this claim
    claim_sources = []
    seen_urls = set()
    
    for s in all_sources:
        if not isinstance(s, dict):
            continue
        
        scid = norm_id(s.get("claim_id"))
        # Match claim-specific sources OR shared sources (no claim_id)
        if scid == cid or scid is None or s.get("claim_id") is None:
            url = s.get("url")
            if url and url not in seen_urls:
                claim_sources.append(s)
                seen_urls.add(url)
    
    # Enrich and assign
    claim_verdict["sources"] = enrich_func(claim_sources)



