"""
Spectrue Scoring Module (M104).

Bayesian credibility scoring with log-odds belief updates.
"""

# Core functions
from spectrue_core.scoring.belief import (
    prob_to_log_odds,
    log_odds_to_prob,
    update_belief,
    apply_consensus_bound,
    calculate_evidence_impact,
    process_updates,
    sigmoid_impact,
)

# RGBA Belief State
from spectrue_core.scoring.rgba_belief import (
    RGBABeliefState,
    create_rgba_belief_from_tier,
)

# Prior calculation
from spectrue_core.scoring.priors import calculate_prior

# Consensus
from spectrue_core.scoring.consensus import calculate_consensus

__all__ = [
    # Functions
    "prob_to_log_odds",
    "log_odds_to_prob",
    "update_belief",
    "apply_consensus_bound",
    "calculate_evidence_impact",
    "process_updates",
    "sigmoid_impact",
    "calculate_prior",
    "calculate_consensus",
    # Classes
    "RGBABeliefState",
    "create_rgba_belief_from_tier",
]
