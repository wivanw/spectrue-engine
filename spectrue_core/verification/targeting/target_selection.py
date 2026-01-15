# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Target selection gate for retrieval optimization.

This module implements the critical gate that selects which claims
get actual Tavily searches vs which are deferred/skipped.

Without this gate, per-claim search causes N×Tavily cost explosion.
The gate ensures only top-K key claims trigger searches.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.orchestration.execution_plan import BudgetClass

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bayesian Target Budget Model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetBudgetParams:
    """
    Parameters for Bayesian target budget computation.
    
    The model computes optimal number of targets by maximizing:
        EVOI(k) - Cost(k)
    
    Where:
        EVOI(k) = sum of expected value of information for k claims
        Cost(k) = marginal_cost_per_target * k
    """
    # Marginal cost per target claim (in normalized units, ~1 Tavily search)
    marginal_cost_per_target: float = 0.5

    # How valuable uncertainty reduction is (domain constant)
    value_uncertainty: float = 1.0

    # Diminishing returns factor: EVOI_i = base * (decay ^ i)
    # Models that later targets have less marginal value
    diminishing_returns_decay: float = 0.85

    # Minimum EVOI threshold to include a target
    min_evoi_threshold: float = 0.05

    # Hard ceiling per budget class (safety net)
    # Maps to BudgetClass enum: MINIMAL, BALANCED, COMPREHENSIVE
    budget_ceilings: dict[str, int] = field(default_factory=lambda: {
        BudgetClass.MINIMAL.value: 3,
        BudgetClass.BALANCED.value: 5,
        BudgetClass.COMPREHENSIVE.value: 30,
    })

    # Hard floor per budget class (minimum targets)
    budget_floors: dict[str, int] = field(default_factory=lambda: {
        BudgetClass.MINIMAL.value: 1,
        BudgetClass.BALANCED.value: 2,
        BudgetClass.COMPREHENSIVE.value: 3,
    })


def _entropy_bernoulli(p: float) -> float:
    """Shannon entropy of Bernoulli(p), normalized to [0,1]."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return h


def _expected_value_of_information(
    claim: dict,
    *,
    prior_p: float = 0.5,
    value_uncertainty: float = 1.0,
) -> float:
    """
    Compute Expected Value of Information (EVOI) for a claim.
    
    EVOI = value_uncertainty * entropy(prior) * claim_worthiness * harm_factor
    
    High EVOI claims:
    - High uncertainty (prior near 0.5)
    - High check_worthiness / importance
    - High harm_potential
    """
    # Entropy captures uncertainty - max at p=0.5
    entropy = _entropy_bernoulli(prior_p)

    # Claim-specific factors
    worthiness = float(claim.get("check_worthiness", claim.get("importance", 0.5)) or 0.5)
    harm = float(claim.get("harm_potential", 0.5) or 0.5)
    # Clamp to 0-1 range
    worthiness = max(0.0, min(1.0, worthiness))
    harm = max(0.0, min(1.0, harm))

    # Confidence inversely affects EVOI (certain claims need less verification)
    conf = claim.get("metadata_confidence")
    if conf:
        conf_str = str(conf).lower()
        if conf_str == "high":
            conf_factor = 0.6
        elif conf_str == "medium":
            conf_factor = 0.8
        else:
            conf_factor = 1.0
    else:
        conf_factor = 1.0

    evoi = value_uncertainty * entropy * worthiness * harm * conf_factor
    return max(0.0, evoi)


def compute_optimal_target_count(
    claims: list[dict],
    *,
    budget_class: str = "minimal",
    params: TargetBudgetParams | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Compute optimal number of targets using Bayesian EVOI model.
    
    Instead of hard-coded limits, we compute:
        k* = argmax_k [ sum_{i=1}^{k} EVOI_i * decay^i - cost * k ]
    
    This naturally:
    - Increases targets when claims have high uncertainty/harm
    - Decreases targets when marginal value drops below cost
    - Respects budget class ceilings as safety nets
    
    Args:
        claims: List of claim dicts with worthiness/harm/confidence
        budget_class: Budget class for ceiling/floor bounds (minimal/standard/deep)
        params: Bayesian model parameters
        
    Returns:
        (optimal_k, trace_dict) where trace_dict contains decision details
    """
    params = params or TargetBudgetParams()

    if not claims:
        return 0, {"reason": "no_claims", "optimal_k": 0}

    n_claims = len(claims)
    ceiling = params.budget_ceilings.get(budget_class, 5)
    floor = params.budget_floors.get(budget_class, 1)

    # Compute EVOI for each claim (already sorted by utility in select_verification_targets)
    evoi_values: list[float] = []
    for claim in claims:
        evoi = _expected_value_of_information(
            claim,
            prior_p=0.5,  # Default prior (uncertain)
            value_uncertainty=params.value_uncertainty,
        )
        evoi_values.append(evoi)

    # Greedy selection: add targets while marginal EVOI > marginal cost
    # With diminishing returns: EVOI_effective(i) = EVOI_i * decay^i
    optimal_k = 0
    cumulative_value = 0.0
    cumulative_cost = 0.0
    marginal_values: list[float] = []

    for i, evoi in enumerate(evoi_values):
        if i >= ceiling:
            break

        # Diminishing returns for later targets
        decay_factor = params.diminishing_returns_decay ** i
        marginal_evoi = evoi * decay_factor
        marginal_cost = params.marginal_cost_per_target

        marginal_values.append({
            "index": i,
            "raw_evoi": evoi,
            "decay_factor": decay_factor,
            "marginal_evoi": marginal_evoi,
            "marginal_cost": marginal_cost,
            "net_value": marginal_evoi - marginal_cost,
        })

        # Stop if marginal value is below threshold or negative
        if marginal_evoi < params.min_evoi_threshold:
            break
        if marginal_evoi < marginal_cost:
            break

        cumulative_value += marginal_evoi
        cumulative_cost += marginal_cost
        optimal_k = i + 1

    # Apply floor constraint
    optimal_k = max(floor, optimal_k)
    # Apply ceiling constraint
    optimal_k = min(ceiling, optimal_k, n_claims)

    trace = {
        "optimal_k": optimal_k,
        "n_claims": n_claims,
        "budget_class": budget_class,
        "ceiling": ceiling,
        "floor": floor,
        "cumulative_value": cumulative_value,
        "cumulative_cost": cumulative_cost,
        "net_value": cumulative_value - cumulative_cost,
        "marginal_analysis": marginal_values[:min(10, len(marginal_values))],
        "params": {
            "marginal_cost_per_target": params.marginal_cost_per_target,
            "value_uncertainty": params.value_uncertainty,
            "diminishing_returns_decay": params.diminishing_returns_decay,
            "min_evoi_threshold": params.min_evoi_threshold,
        },
    }

    Trace.event("target_selection.bayesian_budget", trace)

    return optimal_k, trace


def _assign_semantic_clusters(claims: list[dict], threshold: float = 0.75) -> None:
    """
    Assign cluster_id to claims based on semantic similarity.

    Mutates claims in place, adding 'cluster_id' field.
    Falls back gracefully if embeddings unavailable.
    """
    if not claims:
        return

    try:
        from spectrue_core.utils.embedding_service import EmbedService

        if not EmbedService.is_available():
            return
    except ImportError:
        return

    # Get claim texts
    texts = [c.get("text") or c.get("normalized_text") or "" for c in claims]
    if not any(texts):
        return

    try:
        embeddings = EmbedService.embed(texts, purpose="query")
        if len(embeddings) == 0:
            return

        # Simple greedy clustering: assign to first claim with similarity > threshold
        cluster_ids: list[int] = []
        cluster_reps: list[int] = []  # Index of cluster representative

        for i, vec in enumerate(embeddings):
            assigned = False
            for rep_idx in cluster_reps:
                sim = float(sum(x * y for x, y in zip(vec, embeddings[rep_idx])))
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

        logger.debug(
            "[TargetSelection] Semantic clustering: %d claims -> %d clusters",
            len(claims),
            len(cluster_reps),
        )
    except Exception as e:
        logger.debug("[TargetSelection] Semantic clustering failed: %s", e)


def _tokenize_claim_text(text: str) -> set[str]:
    tokens = re.findall(r"[\w']+", (text or "").lower())
    return {t for t in tokens if t and len(t) > 2}


def _claim_similarity(a: dict, b: dict) -> float:
    text_a = str(a.get("normalized_text") or a.get("text") or "")
    text_b = str(b.get("normalized_text") or b.get("text") or "")
    if not text_a or not text_b:
        return 0.0
    tokens_a = _tokenize_claim_text(text_a)
    tokens_b = _tokenize_claim_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return float(len(inter)) / max(1.0, float(len(union)))


@dataclass
class TargetSelectionResult:
    """Result of target selection gate."""

    # Claims that will get actual Tavily searches
    targets: list[dict] = field(default_factory=list)

    # Claims that won't get searches (will use shared evidence or skip)
    deferred: list[dict] = field(default_factory=list)

    # Mapping: deferred_claim_id -> target_claim_id (for evidence sharing)
    evidence_sharing: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Reason codes for each claim
    reasons: dict[str, str] = field(default_factory=dict)


def select_verification_targets(
    claims: list[dict],
    *,
    max_targets: int | None = None,
    graph_result: Any | None = None,
    budget_class: str = "minimal",
    anchor_claim_id: str | None = None,
    budget_params: TargetBudgetParams | None = None,
) -> TargetSelectionResult:
    """
    Select which claims get actual Tavily searches.

    This is the HARD GATE that prevents per-claim search explosion.
    Uses Bayesian EVOI model to compute optimal number of targets.

    Args:
        claims: All eligible claims
        max_targets: Override for maximum targets (bypasses Bayesian model if set)
        graph_result: Optional claim graph with is_key_claim, centrality, tension
        budget_class: Budget class (minimal/standard/deep)
        anchor_claim_id: If provided (normal profile), this claim MUST be in targets
        budget_params: Parameters for Bayesian target budget model

    Returns:
        TargetSelectionResult with targets and deferred lists
    """
    if not claims:
        return TargetSelectionResult()

    # Use Bayesian EVOI model to compute optimal target count
    # This replaces the previous hard-coded budget_limits heuristic
    budget_params = budget_params or TargetBudgetParams()

    if max_targets is not None:
        # Explicit override - respect user-provided limit but cap at ceiling
        ceiling = budget_params.budget_ceilings.get(budget_class, 5)
        max_targets = min(max_targets, ceiling)
        bayesian_trace = {"mode": "explicit_override", "max_targets": max_targets}
    else:
        # Compute optimal target count using Bayesian EVOI model
        # Note: We need ordered_claims first, so we compute after ordering
        # For now, set to ceiling and refine after ordering
        max_targets, bayesian_trace = compute_optimal_target_count(
            claims,
            budget_class=budget_class,
            params=budget_params,
        )

    # Assign semantic clusters using embeddings (if available)
    _assign_semantic_clusters(claims)

    # Primary ordering comes from EV (harm/uncertainty/centrality/thesis) when graph available
    claim_by_id = {str(c.get("id") or c.get("claim_id") or "c1"): c for c in claims}
    cluster_sizes: dict[str, int] = {}
    for c in claims:
        if not isinstance(c, dict):
            continue
        cluster_id = str(
            c.get("cluster_id")
            or c.get("topic_key")
            or c.get("id")
            or c.get("claim_id")
            or "default"
        )
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    centrality_map: dict[str, float] = {}
    if graph_result and getattr(graph_result, "all_ranked", None):
        ranked_all = list(graph_result.all_ranked)
        max_c = max(
            (float(getattr(r, "centrality_score", 0.0) or 0.0) for r in ranked_all),
            default=1.0,
        )
        for r in ranked_all:
            try:
                centrality_map[r.claim_id] = (
                    float(getattr(r, "centrality_score", 0.0) or 0.0) / max_c
                )
            except Exception:
                continue

    def _conf_score(val):
        if not val:
            return 0.6
        v = str(val).lower()
        if v == "high":
            return 1.0
        if v == "medium":
            return 0.8
        if v == "low":
            return 0.6
        return 0.6

    def _ev(claim_id: str) -> float:
        c = claim_by_id.get(claim_id, {})
        worthiness = float(c.get("check_worthiness", c.get("importance", 0.5)) or 0.5)
        harm = float(c.get("harm_potential", 1) or 1) / 5.0
        uncertainty = 1.0 - _conf_score(c.get("metadata_confidence"))
        centrality = centrality_map.get(claim_id, float(c.get("centrality") or 0.0))
        thesis = (
            1.0
            if str(c.get("claim_role", "core")).lower() in {"core", "thesis"}
            else 0.0
        )
        cost = float(c.get("cost_estimate", 1.0) or 1.0)
        w_h = 0.55
        w_u = 0.20
        w_c = 0.15
        w_t = 0.10
        confidence = _conf_score(c.get("metadata_confidence"))
        signal = w_h * harm + w_u * uncertainty + w_c * centrality + w_t * thesis
        return (signal * worthiness * confidence) / max(cost, 1e-3)

    ordered_ids: list[str] = []
    if graph_result and not getattr(graph_result, "disabled", False):
        ranked = (
            getattr(graph_result, "key_claims", None)
            or getattr(graph_result, "all_ranked", None)
            or []
        )
        ordered_ids = [r.claim_id for r in ranked if getattr(r, "claim_id", None)]
        ordered_ids = sorted(ordered_ids, key=lambda cid: _ev(cid), reverse=True)

    seen: set[str] = set()
    ordered_claims: list[dict] = []
    for cid in ordered_ids:
        if cid in claim_by_id and cid not in seen:
            ordered_claims.append(claim_by_id[cid])
            seen.add(cid)
    for cid, c in claim_by_id.items():
        if cid not in seen:
            ordered_claims.append(c)
            seen.add(cid)

    # ANCHOR GUARANTEE: If anchor_claim_id is specified (normal profile),
    # ensure the anchor is ALWAYS in targets, even if not in top-K by EV.
    # This fixes the normal_pipeline.violation when anchor is deferred.
    anchor_forced = False
    if anchor_claim_id and str(anchor_claim_id) in claim_by_id:
        anchor_in_top_k = any(
            str(c.get("id") or c.get("claim_id")) == str(anchor_claim_id)
            for c in ordered_claims[:max_targets]
        )
        if not anchor_in_top_k:
            # Find original rank before reordering
            original_rank = next(
                (
                    i
                    for i, c in enumerate(ordered_claims)
                    if str(c.get("id") or c.get("claim_id")) == str(anchor_claim_id)
                ),
                -1,
            )
            # Move anchor to the front of ordered_claims
            anchor_claim = claim_by_id[str(anchor_claim_id)]
            ordered_claims = [anchor_claim] + [
                c
                for c in ordered_claims
                if str(c.get("id") or c.get("claim_id")) != str(anchor_claim_id)
            ]
            anchor_forced = True
            Trace.event(
                "target_selection.anchor_forced",
                {
                    "anchor_claim_id": str(anchor_claim_id),
                    "reason": "anchor_not_in_top_k",
                    "original_rank": original_rank,
                },
            )

    # Split into targets and deferred using deterministic ordered_claims
    result = TargetSelectionResult()
    cluster_map: dict[str, str] = {}  # cluster_id -> target_claim_id

    for i, claim in enumerate(ordered_claims):
        claim_id = str(claim.get("id") or claim.get("claim_id") or "c1")
        cluster_id = str(
            claim.get("cluster_id")
            or claim.get("topic_key")
            or claim.get("id")
            or claim.get("claim_id")
            or "default"
        )

        if i < max_targets:
            # This claim gets actual search
            result.targets.append(claim)
            result.reasons[claim_id] = f"target_rank_{i + 1}"

            # Mark as cluster representative
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = claim_id
        else:
            # Deferred - won't trigger search
            result.deferred.append(claim)

            # Check if there's a target in same cluster for evidence sharing
            if cluster_id in cluster_map:
                target_id = cluster_map[cluster_id]
                target_claim = claim_by_id.get(target_id, {})
                similarity = (
                    _claim_similarity(claim, target_claim) if target_claim else 0.0
                )
                cluster_size = max(1, cluster_sizes.get(cluster_id, 1))
                cohesion = min(1.0, 1.0 / (cluster_size**0.5))
                result.evidence_sharing[claim_id] = {
                    "target_id": target_id,
                    "similarity": similarity,
                    "cohesion": cohesion,
                }
                result.reasons[claim_id] = f"deferred_shares_{target_id}"
            elif cluster_map:
                # Fallback - share with first target when no cluster match
                # Better than leaving claim without any verdict
                first_target = next(iter(cluster_map.values()))
                target_claim = claim_by_id.get(first_target, {})
                similarity = (
                    _claim_similarity(claim, target_claim) if target_claim else 0.0
                )
                result.evidence_sharing[claim_id] = {
                    "target_id": first_target,
                    "similarity": similarity,
                    "cohesion": 0.0,
                }
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
            "anchor_claim_id": str(anchor_claim_id) if anchor_claim_id else None,
            "anchor_forced": anchor_forced,
            "bayesian_trace": bayesian_trace,
            "target_ids": [
                str(cid)
                for cid in (c.get("id") or c.get("claim_id") for c in result.targets)
                if cid is not None
            ],
            "deferred_ids": [
                str(cid)
                for cid in (c.get("id") or c.get("claim_id") for c in result.deferred)
                if cid is not None
            ],
        },
    )

    return result


def propagate_deferred_verdicts(
    result: dict,
    evidence_sharing: dict[str, Any],
    deferred_claims: list[dict],
) -> dict:
    """
    Propagate verdicts from target claims to deferred claims.

    Deferred claims inherit verdicts from their target claim (same cluster)
    with a penalty applied and reason marked as "derived".

    Args:
        result: Evidence flow result containing claim_verdicts
        evidence_sharing: Map of deferred_claim_id -> share payload
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

        share = evidence_sharing.get(claim_id)
        if not share:
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

        if isinstance(share, dict):
            target_id = str(share.get("target_id") or "")
            similarity = float(share.get("similarity") or 0.0)
            cohesion = float(share.get("cohesion") or 0.0)
        else:
            target_id = str(share)
            similarity = 0.0
            cohesion = 0.0

        if not target_id:
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

        # Minimum similarity threshold for verdict inheritance
        # Claims with low similarity should NOT copy sources/reason from unrelated targets
        MIN_INHERITANCE_SIMILARITY = 0.25
        if similarity < MIN_INHERITANCE_SIMILARITY:
            # Too dissimilar - mark as NEI, don't copy sources/reason
            payload = {
                "claim_id": claim_id,
                "verdict_score": 0.5,
                "verdict": "insufficient_evidence",
                "verdict_state": "insufficient_evidence",
                "derived_from": target_id,
                "derived_reason": "similarity_below_threshold",
                "reasons_short": [f"Claim too dissimilar to target (similarity={similarity:.2f} < {MIN_INHERITANCE_SIMILARITY})"],
                "propagation": {
                    "similarity": similarity,
                    "cohesion": cohesion,
                    "mode": "rejected_low_similarity",
                    "threshold": MIN_INHERITANCE_SIMILARITY,
                },
            }
            if claim_id in verdict_index:
                claim_verdicts[verdict_index[claim_id]] = payload
            else:
                claim_verdicts.append(payload)
            propagated_count += 1
            continue

        # Inherit directly from target (no shrinkage/postprocessing per Spec)
        # "Deep analysis returns per-claim results... no backend postprocessing"
        target_score = float(target_cv.get("verdict_score", 0.5) or 0.5)
        derived_score = target_score

        payload = {
            "claim_id": claim_id,
            "verdict_score": derived_score,
            "verdict": target_cv.get("verdict", "ambiguous"),
            "verdict_state": target_cv.get("verdict_state", "insufficient_evidence"),
            "derived_from": target_id,
            "derived_reason": "inherited_from_cluster_target",
            "reasons_short": ["Verdict derived from key claim in same cluster"],
            "propagation": {
                "similarity": similarity,
                "cohesion": cohesion,
                "mode": "direct_copy",
            },
            # Inherit sources and reason
            "sources": target_cv.get("sources", []),
            "reason": f"(Derived from {target_id}) " + (target_cv.get("reason") or ""),
            # Copy RGBA directly (same verdict = same color)
            "rgba": target_cv.get("rgba"),
        }

        if claim_id in verdict_index:
            claim_verdicts[verdict_index[claim_id]] = payload
        else:
            claim_verdicts.append(payload)
        propagated_count += 1

    sharing_sample = list(evidence_sharing.items())[:5]
    Trace.event(
        "deferred_verdicts.propagated",
        {
            "propagated_count": propagated_count,
            "evidence_sharing_sample": sharing_sample,
        },
    )

    return result
