# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Target selection gate for retrieval optimization.

This module implements the critical gate that selects which claims
get actual Tavily searches vs which are deferred/skipped.

Without this gate, per-claim search causes N×Tavily cost explosion.
The gate ensures only top-K key claims trigger searches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def _assign_semantic_clusters(claims: list[dict], threshold: float = 0.75) -> None:
    """
    M109: Assign cluster_id to claims based on semantic similarity.
    
    Mutates claims in place, adding 'cluster_id' field.
    Falls back gracefully if embeddings unavailable.
    """
    if not claims:
        return
    
    try:
        from spectrue_core.embeddings import EmbedService
        if not EmbedService.is_available():
            return
    except ImportError:
        return
    
    # Get claim texts
    texts = [c.get("text") or c.get("normalized_text") or "" for c in claims]
    if not any(texts):
        return
    
    try:
        import numpy as np
        embeddings = EmbedService.embed(texts)
        if len(embeddings) == 0:
            return
        
        # Simple greedy clustering: assign to first claim with similarity > threshold
        cluster_ids: list[int] = []
        cluster_reps: list[int] = []  # Index of cluster representative
        
        for i, vec in enumerate(embeddings):
            assigned = False
            for rep_idx in cluster_reps:
                sim = float(np.dot(vec, embeddings[rep_idx]))
                if sim >= threshold:
                    cluster_ids.append(cluster_ids[rep_idx])
                    assigned = True
                    break
            
            if not assigned:
                # New cluster
                new_cluster = len(cluster_reps)
                cluster_ids.append(new_cluster)
                cluster_reps.append(i)
        
        # Assign to claims
        for claim, cid in zip(claims, cluster_ids):
            claim["cluster_id"] = f"semantic_{cid}"
        
        logger.debug("[TargetSelection] Semantic clustering: %d claims -> %d clusters", 
                     len(claims), len(cluster_reps))
    except Exception as e:
        logger.debug("[TargetSelection] Semantic clustering failed: %s", e)


@dataclass
class TargetSelectionResult:
    """Result of target selection gate."""
    
    # Claims that will get actual Tavily searches
    targets: list[dict] = field(default_factory=list)
    
    # Claims that won't get searches (will use shared evidence or skip)
    deferred: list[dict] = field(default_factory=list)
    
    # Mapping: deferred_claim_id -> target_claim_id (for evidence sharing)
    evidence_sharing: dict[str, str] = field(default_factory=dict)
    
    # Reason codes for each claim
    reasons: dict[str, str] = field(default_factory=dict)


def select_verification_targets(
    claims: list[dict],
    *,
    max_targets: int = 2,
    graph_result: Any | None = None,
    budget_class: str = "minimal",
) -> TargetSelectionResult:
    """
    Select which claims get actual Tavily searches.
    
    This is the HARD GATE that prevents per-claim search explosion.
    Only top-K claims (based on graph metrics + worthiness) get searches.
    
    Args:
        claims: All eligible claims
        max_targets: Maximum claims that can trigger searches (default=2)
        graph_result: Optional claim graph with is_key_claim, centrality, tension
        budget_class: Budget class (minimal/standard/deep)
        
    Returns:
        TargetSelectionResult with targets and deferred lists
    """
    if not claims:
        return TargetSelectionResult()
    
    # Adjust max_targets based on budget_class
    if budget_class == "minimal":
        max_targets = min(max_targets, 2)
    elif budget_class == "standard":
        max_targets = min(max_targets, 3)
    elif budget_class == "deep":
        max_targets = min(max_targets, 5)
    
    # M109: Assign semantic clusters using embeddings (if available)
    _assign_semantic_clusters(claims)
    
    # Score each claim for target selection priority
    scored_claims: list[tuple[float, dict]] = []
    
    for claim in claims:
        claim_id = str(claim.get("id") or claim.get("claim_id") or "c1")
        
        # Base score from check_worthiness (0-1)
        worthiness = float(claim.get("check_worthiness", 0.5) or 0.5)
        
        # Importance boost
        importance = float(claim.get("importance", 0.5) or 0.5)
        
        # Graph-based boosts
        graph_score = 0.0
        is_key = False
        if graph_result:
            # Check if this is a key claim from graph analysis
            key_ids = getattr(graph_result, "key_claim_ids", None) or []
            is_key = claim_id in key_ids
            if is_key:
                graph_score += 0.3
            
            # Centrality boost
            ranked = getattr(graph_result, "ranked", None) or []
            for r in ranked:
                if r.claim_id == claim_id:
                    graph_score += float(r.structural_importance or 0) * 0.2
                    break
            
            # Tension boost (controversial claims need more scrutiny)
            tension = float(claim.get("graph_tension_score", 0) or 0)
            graph_score += tension * 0.1
        
        # Claim type boost (thesis > support > background)
        # M108: Thesis claims get massive boost to ALWAYS be in targets
        type_boost = 0.0
        claim_type = str(claim.get("type", "support")).lower()
        claim_role = str(claim.get("claim_role", "support")).lower()
        is_thesis = claim_type == "core" or claim_role == "thesis"
        if is_thesis:
            type_boost = 10.0  # M108: Guarantee thesis is first
        elif claim_type == "support":
            type_boost = 0.1
        
        # Final score
        score = worthiness * 0.4 + importance * 0.3 + graph_score + type_boost
        scored_claims.append((score, claim))
    
    # Sort by score descending
    scored_claims.sort(key=lambda x: x[0], reverse=True)
    
    # Split into targets and deferred
    result = TargetSelectionResult()
    cluster_map: dict[str, str] = {}  # cluster_id -> target_claim_id
    
    for i, (score, claim) in enumerate(scored_claims):
        claim_id = str(claim.get("id") or claim.get("claim_id") or "c1")
        cluster_id = str(claim.get("cluster_id") or claim.get("topic_key") or "default")
        
        if i < max_targets:
            # This claim gets actual search
            result.targets.append(claim)
            result.reasons[claim_id] = f"target_rank_{i+1}"
            
            # Mark as cluster representative
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = claim_id
        else:
            # Deferred - won't trigger search
            result.deferred.append(claim)
            
            # Check if there's a target in same cluster for evidence sharing
            if cluster_id in cluster_map:
                result.evidence_sharing[claim_id] = cluster_map[cluster_id]
                result.reasons[claim_id] = f"deferred_shares_{cluster_map[cluster_id]}"
            elif cluster_map:
                # M108: Fallback - share with first target when no cluster match
                # Better than leaving claim without any verdict
                first_target = next(iter(cluster_map.values()))
                result.evidence_sharing[claim_id] = first_target
                result.reasons[claim_id] = f"deferred_shares_{first_target}_fallback"
            else:
                result.reasons[claim_id] = "deferred_no_search"
    
    Trace.event(
        "target_selection.completed",
        {
            "total_claims": len(claims),
            "targets_count": len(result.targets),
            "deferred_count": len(result.deferred),
            "evidence_sharing_count": len(result.evidence_sharing),
            "max_targets": max_targets,
            "budget_class": budget_class,
            "target_ids": [c.get("id") for c in result.targets],
            "deferred_ids": [c.get("id") for c in result.deferred],
        },
    )
    
    return result


def propagate_deferred_verdicts(
    result: dict,
    evidence_sharing: dict[str, str],
    deferred_claims: list[dict],
) -> dict:
    """
    Propagate verdicts from target claims to deferred claims.
    
    Deferred claims inherit verdicts from their target claim (same cluster)
    with a penalty applied and reason marked as "derived".
    
    Args:
        result: Evidence flow result containing claim_verdicts
        evidence_sharing: Map of deferred_claim_id -> target_claim_id
        deferred_claims: List of deferred claim dicts
        
    Returns:
        Updated result with verdicts for deferred claims
    """
    if not evidence_sharing or not deferred_claims:
        return result
    
    claim_verdicts = result.get("claim_verdicts")
    if not isinstance(claim_verdicts, list):
        return result
    
    # Build lookup: target_claim_id -> verdict
    target_verdicts: dict[str, dict] = {}
    verdict_index: dict[str, int] = {}
    for idx, cv in enumerate(claim_verdicts):
        if isinstance(cv, dict):
            cid = str(cv.get("claim_id") or "")
            if cid:
                target_verdicts[cid] = cv
                verdict_index[cid] = idx
    
    # Propagate to deferred claims
    propagated_count = 0
    for claim in deferred_claims:
        claim_id = str(claim.get("id") or claim.get("claim_id") or "")
        if not claim_id:
            continue
        
        target_id = evidence_sharing.get(claim_id)
        if not target_id:
            # No target to inherit from - mark as NEI
            payload = {
                "claim_id": claim_id,
                "verdict_score": 0.5,
                "verdict": "insufficient_evidence",
                "verdict_state": "insufficient_evidence",
                "derived_from": None,
                "derived_reason": "no_target_in_cluster",
            }
            if claim_id in verdict_index:
                claim_verdicts[verdict_index[claim_id]] = payload
            else:
                claim_verdicts.append(payload)
            propagated_count += 1
            continue
        
        target_cv = target_verdicts.get(target_id)
        if not target_cv:
            # Target has no verdict
            payload = {
                "claim_id": claim_id,
                "verdict_score": 0.5,
                "verdict": "insufficient_evidence",
                "verdict_state": "insufficient_evidence",
                "derived_from": target_id,
                "derived_reason": "target_no_verdict",
            }
            if claim_id in verdict_index:
                claim_verdicts[verdict_index[claim_id]] = payload
            else:
                claim_verdicts.append(payload)
            propagated_count += 1
            continue
        
        # Inherit from target with penalty
        target_score = float(target_cv.get("verdict_score", 0.5) or 0.5)
        # Apply 10% penalty toward 0.5 (uncertainty)
        derived_score = target_score * 0.9 + 0.5 * 0.1
        
        payload = {
            "claim_id": claim_id,
            "verdict_score": derived_score,
            "verdict": target_cv.get("verdict", "ambiguous"),
            "verdict_state": target_cv.get("verdict_state", "insufficient_evidence"),
            "derived_from": target_id,
            "derived_reason": "inherited_from_cluster_target",
            "reasons_short": ["Verdict derived from key claim in same cluster"],
        }
        if claim_id in verdict_index:
            claim_verdicts[verdict_index[claim_id]] = payload
        else:
            claim_verdicts.append(payload)
        propagated_count += 1
    
    Trace.event(
        "deferred_verdicts.propagated",
        {
            "propagated_count": propagated_count,
            "evidence_sharing_map": evidence_sharing,
        },
    )
    
    return result
