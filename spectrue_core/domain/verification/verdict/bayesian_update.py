"""Bayesian belief update functions for credibility scoring."""

from __future__ import annotations

from .model import (
    BeliefState,
    ConsensusState,
    prob_to_log_odds,
    log_odds_to_prob,
    update_belief,
    process_updates,
    calculate_evidence_impact,
    sigmoid_impact,
)


__all__ = [
    "update_belief",
    "apply_consensus_bound",
    "calculate_evidence_impact",
    "process_updates",
    "sigmoid_impact",
    "prob_to_log_odds",
    "log_odds_to_prob",
]


def apply_consensus_bound(belief: BeliefState, consensus: ConsensusState) -> BeliefState:
    """
    Bounds the posterior belief using the Scientific Consensus latent variable.
    The credibility cannot exceed the scientific consensus on the topic.
    """
    if consensus.source_count < 2:
        return belief

    limit_log_odds = prob_to_log_odds(consensus.score)
    new_log_odds = min(belief.log_odds, limit_log_odds)

    return BeliefState(log_odds=new_log_odds, confidence=belief.confidence)