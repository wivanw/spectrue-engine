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




def assign_claim_rgba(
    claim_verdict: dict[str, Any],
    *,
    global_r: float,
    global_b: float,
    global_a: float,
    judge_mode: ScoringMode | str,
    pack: dict[str, Any] | None = None,
) -> None:
    """
    Assign RGBA to claim verdict based on mode.
    
    M119 Unified Logic:
    1. Extract A_llm (from judge output or global fallback)
    2. Compute A_det (deterministic fallback based on evidence reliability)
    3. Compute A_cap (safety ceiling based on diversity and anchors)
    4. A_final = min(A_llm, A_cap) if A_llm exists, else min(A_det, A_cap)
    
    Args:
        claim_verdict: Claim verdict dict (mutated in-place)
        global_r: Global danger score
        global_b: Global bias score
        global_a: Global explainability score
        judge_mode: ScoringMode enum or string
        pack: Full EvidencePack (M119)
    """
    from spectrue_core.utils.trace import Trace
    from spectrue_core.utils.evidence_alpha import compute_A_det
    from spectrue_core.domain.verification.stance.aggregation import apply_conflict_explainability_penalty
    from spectrue_core.pipeline.mode import ScoringMode
    
    cid = claim_verdict.get("claim_id", "unknown")
    g_score = float(claim_verdict.get("verdict_score", 0.5) or 0.5)
    existing_rgba = claim_verdict.get("rgba")
    
    # Handle enum or string input
    mode_value = judge_mode.value if isinstance(judge_mode, ScoringMode) else judge_mode
    
    from spectrue_core.domain.evidence.model import norm_claim_id
    target_cid = norm_claim_id(cid)
    
    has_valid_rgba = (
        isinstance(existing_rgba, list)
        and len(existing_rgba) == 4
        and all(isinstance(x, (int, float)) for x in existing_rgba)
    )
    
    # 1. Base scores
    r_val = global_r
    g_val = g_score
    b_val = global_b
    a_llm = None
    
    if has_valid_rgba:
        r_val, g_val, b_val, a_llm = existing_rgba
    elif global_a >= 0:
        a_llm = global_a

    # 2. Extract from verdict_payload as second priority (Deep result storage)
    if a_llm is None and claim_verdict.get("verdict_payload"):
        v_payload = claim_verdict["verdict_payload"]
        if isinstance(v_payload, dict):
            v_rgba = v_payload.get("rgba")
            if isinstance(v_rgba, list) and len(v_rgba) == 4:
                a_llm = v_rgba[3]

    # 3. Handle explicit DEEP mode errors
    if mode_value == ScoringMode.DEEP.value and not has_valid_rgba:
        if claim_verdict.get("status") == "error":
            Trace.event("cv.rgba_assigned", {"claim_id": cid, "judge_mode": mode_value, "source": "error", "rgba": None})
            return
        # If not error but missing RGBA, we proceed to compute A_det as fallback below
        # but log it as missing
        Trace.event("cv.rgba_missing_in_deep", {"claim_id": cid})

    # 4. M133: Alpha — LLM score passes through unchanged (no cap)
    items = pack.get("items", []) if isinstance(pack, dict) else []
    
    # A_det (fallback only)
    a_det = compute_A_det(items, cid)
    
    claim_items = []
    for it in items:
        item_cid = norm_claim_id(it.get("claim_id"))
        if item_cid is not None and item_cid != target_cid:
            continue
        claim_items.append(it)

    # Count for conflict penalty
    support_with_quote = [
        it for it in claim_items 
        if it.get("quote") and (it.get("stance") or "").upper() == "SUPPORT"
    ]
    refute_with_quote = [
        it for it in claim_items 
        if it.get("quote") and (it.get("stance") or "").upper() == "REFUTE"
    ]
    
    # A_final: LLM score if available, else deterministic (no cap)
    if a_llm is not None:
        a_final = float(a_llm)
        a_source = "llm" if has_valid_rgba else "global"
    else:
        a_final = a_det
        a_source = "det"
        
    # 5. Conflict Penalty (M119 Rule: Requires both SUPPORT and REFUTE with direct anchors)
    if len(support_with_quote) > 0 and len(refute_with_quote) > 0:
        a_final = apply_conflict_explainability_penalty(a_final)
        a_source += "_with_conflict_penalty"
    
    claim_verdict["rgba"] = [float(r_val), float(g_val), float(b_val), float(a_final)]
    
    Trace.event(
        "cv.rgba_assigned",
        {
            "claim_id": cid,
            "judge_mode": mode_value,
            "source": a_source,
            "rgba": claim_verdict["rgba"],
            "a_det": a_det,
            "a_llm": a_llm,
        },
    )


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
    from spectrue_core.utils.evidence_scoring import norm_id
    
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

