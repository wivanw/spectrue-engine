# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence Verdict Processing

Processing and enrichment of claim verdicts from LLM scoring.
Extracted from pipeline_evidence.py for better modularity.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from spectrue_core.verification.evidence.evidence_stance import (
    count_stance_evidence,
    derive_verdict_from_score,
    derive_verdict_state_from_llm_score,
    detect_evidence_conflict,
    check_has_direct_evidence,
    CANONICAL_VERDICT_STATES,
    assign_claim_rgba,
    enrich_claim_sources,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.pipeline.mode import ScoringMode

# M133: compute_explainability_tier_adjustment removed
from spectrue_core.verification.evidence.evidence_scoring import norm_id as _norm_id

logger = logging.getLogger(__name__)


def process_single_claim_verdict(
    cv: dict[str, Any],
    claim_id: str,
    claim_obj: dict[str, Any] | None,
    temporality: Any,
    has_direct_evidence: bool,
    pack: dict[str, Any],
    explainability: float,
) -> dict[str, Any]:
    """
    Process a single claim verdict with evidence stats and explainability.
    
    Args:
        cv: Claim verdict dict (mutated in-place)
        claim_id: Normalized claim ID
        claim_obj: Full claim object
        temporality: Claim temporality metadata
        has_direct_evidence: Whether claim has quoted evidence
        pack: Evidence pack
        explainability: Global explainability score
        
    Returns:
        Processing result dict with verdict_state, veracity_entry, etc.
    """
    # Get LLM verdict score (raw observation)
    llm_score = cv.get("verdict_score")
    if not isinstance(llm_score, (int, float)):
        llm_score = 0.5
    llm_score = float(llm_score)

    # Count stance evidence using extracted function
    items = pack.get("items", []) if isinstance(pack, dict) else []
    n_support, n_refute, best_tier = count_stance_evidence(claim_id, items)

    # Use raw LLM score directly (no posterior boost)
    # Posterior was causing all scores to drift toward 0.98+
    cv["verdict_score"] = llm_score

    # Derive verdict from LLM score using extracted function
    cv["verdict"] = derive_verdict_from_score(llm_score)
    cv["status"] = cv["verdict"]

    # Legacy reasons_expert for backward compatibility
    cv["reasons_expert"] = {
        "best_tier": best_tier,
        "n_support": n_support,
        "n_refute": n_refute,
    }
    cv["reasons_short"] = cv.get("reasons_short", []) or []

    explainability_update = None

    veracity_entry = {
        "claim_id": claim_id,
        "role": claim_obj.get("claim_role") if claim_obj else None,
        "target": claim_obj.get("verification_target") if claim_obj else None,
        "harm": claim_obj.get("harm_potential") if claim_obj else None,
        "llm_verdict": cv.get("verdict"),
        "llm_score": llm_score,
        "has_direct_evidence": has_direct_evidence,
        "best_tier": best_tier,
    }

    # Verdict state derivation using extracted function
    existing_state = str(cv.get("verdict_state") or "").lower().strip()

    if existing_state in CANONICAL_VERDICT_STATES:
        verdict_state = existing_state
    else:
        verdict_state = derive_verdict_state_from_llm_score(
            llm_score, n_support, n_refute
        )

    cv["verdict_state"] = verdict_state

    # Conflict detection using extracted function
    has_conflict = detect_evidence_conflict(n_support, n_refute)

    # M119: local_explainability is deprecated; handled in assign_claim_rgba

    return {
        "claim_id": claim_id,
        "verdict_state": verdict_state,
        "veracity_entry": veracity_entry,
        "explainability_update": explainability_update,
        "has_conflict": has_conflict,
        "llm_score": llm_score,
    }


def process_claim_verdicts(
    claim_verdicts: list[dict[str, Any]],
    claims: list[dict],
    pack: dict[str, Any],
    explainability: float,
) -> tuple[dict[str, str], list[dict], bool, float | None]:
    """
    Process all claim verdicts in parallel.
    
    Args:
        claim_verdicts: List of claim verdict dicts
        claims: List of claim objects
        pack: Evidence pack
        explainability: Global explainability score
        
    Returns:
        Tuple of (verdict_state_by_claim, veracity_debug, conflict_detected, explainability_update)
    """
    verdict_state_by_claim: dict[str, str] = {}
    veracity_debug: list[dict] = []
    conflict_detected = False
    explainability_final = None
    
    # Prepare all CV data
    cv_data_list = []
    items = pack.get("items", []) if isinstance(pack, dict) else []

    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        
        claim_id = _norm_id(cv.get("claim_id"))
        if not claim_id and claims:
            claim_id = _norm_id(claims[0].get("id") or "c1")
            cv["claim_id"] = claim_id

        claim_obj = None
        for c in claims or []:
            if _norm_id(c.get("id") or c.get("claim_id")) == claim_id:
                claim_obj = c
                break
        
        temporality = (
            claim_obj.get("temporality") if isinstance(claim_obj, dict) else None
        )

        # Check for direct evidence using extracted function
        has_direct_evidence = check_has_direct_evidence(claim_id, items)

        cv_data_list.append({
            "cv": cv,
            "claim_id": claim_id,
            "claim_obj": claim_obj,
            "temporality": temporality,
            "has_direct_evidence": has_direct_evidence,
            "pack": pack,
            "explainability": explainability,
        })

    # Process in parallel using thread pool
    if len(cv_data_list) > 1:
        # Parallel execution for multiple claims
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(cv_data_list))
        ) as executor:
            results_list = list(executor.map(
                lambda d: process_single_claim_verdict(**d), 
                cv_data_list
            ))
    else:
        # Sequential for single claim (avoid thread overhead)
        results_list = [process_single_claim_verdict(**d) for d in cv_data_list]

    # Merge results
    for res in results_list:
        verdict_state_by_claim[res["claim_id"]] = res["verdict_state"]
        veracity_debug.append(res["veracity_entry"])
        if res["explainability_update"] is not None:
            explainability_final = res["explainability_update"]
        if res["has_conflict"]:
            conflict_detected = True

    return verdict_state_by_claim, veracity_debug, conflict_detected, explainability_final


def enrich_all_claim_verdicts(
    claim_verdicts: list[dict[str, Any]],
    pack: dict[str, Any],
    enrich_sources_with_trust: Any,
    global_rgba: tuple[float, float, float, float],
    scoring_mode: "ScoringMode" | str,
) -> None:
    """
    Enrich all claim verdicts with sources and RGBA.
    
    Args:
        claim_verdicts: List of claim verdict dicts (mutated in-place)
        pack: Evidence pack with all sources
        enrich_sources_with_trust: Function to enrich sources
        global_rgba: Tuple of (R, G, B, A) global scores
        scoring_mode: ScoringMode.STANDARD or ScoringMode.DEEP (or string for compatibility)
    """
    all_scored = pack.get("scored_sources") or []
    all_context = pack.get("context_sources") or []
    all_sources = all_scored + all_context
    
    global_r, global_g, global_b, global_a = global_rgba

    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue

        # Enrich per-claim sources using extracted function
        enrich_claim_sources(cv, all_sources, enrich_sources_with_trust)

        # Assign RGBA using extracted function
        assign_claim_rgba(
            cv,
            global_r=global_r,
            global_b=global_b,
            global_a=global_a,
            judge_mode=scoring_mode,
            pack=pack,
        )

