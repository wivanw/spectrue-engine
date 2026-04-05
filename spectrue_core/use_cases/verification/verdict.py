"""Use cases for verification verdict and Bayesian belief updates."""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.domain.verification.verdict.model import BeliefState
from spectrue_core.adapters.graph.context import ClaimContextGraph
from spectrue_core.domain.verification.verdict.propagation import (
    propagate_belief,
    propagation_routing_signals,
)
from spectrue_core.domain.verification.verdict.bayesian_update import (
    calculate_evidence_impact,
    update_belief,
    log_odds_to_prob,
    apply_consensus_bound,
)
from spectrue_core.domain.verification.verdict.consensus import calculate_consensus
from spectrue_core.utils.trace import Trace

from spectrue_core.utils.evidence_scoring import (
    norm_id as _norm_id,
    is_prob as _is_prob,
    compute_article_g_from_anchor as _compute_article_g_from_anchor,
    select_anchor_for_article_g as _select_anchor_for_article_g,
)

logger = logging.getLogger(__name__)


class MockEvidence:
    """Wrapper to adapt dict evidence to object interface for consensus calculation."""
    
    def __init__(self, d: dict):
        self.domain = d.get("domain")
        self.stance = d.get("stance")


def apply_bayesian_update(
    *,
    prior_belief: BeliefState,
    context_graph: ClaimContextGraph | None,
    claim_verdicts: list[dict],
    anchor_claim_id: str | None,
    importance_by_claim: dict[str, float],
    veracity_debug: list[dict],
    pack: Any,  # Evidence pack (dict or object)
    result: dict,  # Mutated in place
    raw_verified_score: float | None,
    raw_confidence_score: float | None,
    raw_rationale: str | None,
) -> dict:
    """
    Apply Bayesian belief update from claim verdicts.
    """
    current_belief = prior_belief
    
    # Consensus Calculation
    evidence_list = []
    raw_evidence = (
        getattr(pack, "evidence", [])
        if not isinstance(pack, dict)
        else pack.get("evidence", [])
    )
    
    for e in raw_evidence:
        if isinstance(e, dict):
            evidence_list.append(MockEvidence(e))
        else:
            evidence_list.append(e)
    
    consensus = calculate_consensus(evidence_list)
    
    # Claim Graph Propagation
    if context_graph and isinstance(claim_verdicts, list):
        for cv in claim_verdicts:
            cid = cv.get("claim_id")
            node = context_graph.get_node(cid)
            if node:
                v = cv.get("verdict", "ambiguous")
                conf = cv.get("confidence")
                if not _is_prob(conf):
                    conf = cv.get("verdict_score")
                if not _is_prob(conf):
                    conf = 0.5
                impact = calculate_evidence_impact(v, confidence=conf)
                node.local_belief = BeliefState(log_odds=impact)
        
        propagate_belief(context_graph)
        result["graph_propagation"] = propagation_routing_signals(context_graph)
        
        # Update from Anchor
        if anchor_claim_id:
            anchor_node = context_graph.get_node(anchor_claim_id)
            if anchor_node and anchor_node.propagated_belief:
                current_belief = update_belief(
                    current_belief, anchor_node.propagated_belief.log_odds
                )
    
    elif isinstance(claim_verdicts, list):
        # Fallback: Sum updates (weighted by verdict strength + claim importance)
        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            v = cv.get("verdict", "ambiguous")
            cid = _norm_id(cv.get("claim_id"))
            try:
                strength = float(cv.get("verdict_score", 0.5) or 0.5)
            except Exception:
                strength = 0.5
            strength = max(0.0, min(1.0, strength))
            relevance = max(0.0, min(1.0, importance_by_claim.get(cid, 1.0)))
            impact = calculate_evidence_impact(
                v, confidence=strength, relevance=relevance
            )
            current_belief = update_belief(current_belief, impact)
    
    # Apply Consensus
    current_belief = apply_consensus_bound(current_belief, consensus)
    belief_g = log_odds_to_prob(current_belief.log_odds)
    
    # Select anchor for article G
    anchor_for_g = anchor_claim_id
    anchor_dbg = {}
    if isinstance(claim_verdicts, list) and veracity_debug:
        anchor_for_g, anchor_dbg = _select_anchor_for_article_g(
            anchor_claim_id=anchor_claim_id,
            claim_verdicts=claim_verdicts,
            veracity_debug=veracity_debug,
        )
        Trace.event("anchor_selection.post_evidence", anchor_dbg)
    
    # Article-level G: pure anchor formula
    g_article, g_dbg = _compute_article_g_from_anchor(
        anchor_claim_id=anchor_for_g,
        claim_verdicts=claim_verdicts if isinstance(claim_verdicts, list) else None,
        prior_p=0.5,
    )
    prev = result.get("verified_score")
    result["verified_score"] = g_article
    Trace.event(
        "verdict.article_g_formula",
        {
            **g_dbg,
            "prev_verified_score": prev,
            "belief_score": belief_g,
        },
    )
    
    Trace.event_full(
        "verdict.veracity_debug",
        {
            "raw_verified_score": raw_verified_score,
            "raw_confidence_score": raw_confidence_score,
            "final_verified_score": result.get("verified_score"),
            "final_confidence_score": result.get("confidence_score"),
            "veracity_by_claim": veracity_debug,
            "rationale": raw_rationale,
        },
    )
    
    # Return trace
    return {
        "prior_log_odds": prior_belief.log_odds,
        "consensus_score": consensus.score,
        "posterior_log_odds": current_belief.log_odds,
        "final_probability": result["verified_score"],
        "belief_probability": belief_g,
    }
