# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
    "calculate_consensus",
    # Classes
    "RGBABeliefState",
    "create_rgba_belief_from_tier",
]
