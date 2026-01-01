# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Bayesian belief update functions for credibility scoring.

This module provides pure functions for log-odds conversion and belief updates.
For RGBA belief state class, see `spectrue_core.scoring.rgba_belief`.

Mathematical Foundation:
------------------------
Log-odds (logit) representation makes Bayesian updates additive:
    Posterior(LO) = Prior(LO) + Evidence(LO)

Reference: Good, I.J. (1950). Probability and the Weighing of Evidence.
"""

import math
from typing import List

from spectrue_core.schema.scoring import BeliefState, ConsensusState


def prob_to_log_odds(p: float, epsilon: float = 1e-9) -> float:
    """
    Convert probability to log-odds (logit function).
    Clips probability to [epsilon, 1-epsilon] to avoid infinity.
    
    Mathematical rationale:
    -----------------------
    Log-odds is defined as: log_odds = log(p / (1 - p))
    
    Properties:
    1. Additivity: Posterior(LO) = Prior(LO) + log(Likelihood Ratio)
    2. Unbounded range: [-∞, +∞] vs probability's [0, 1]
    3. Symmetry: log_odds(p) = -log_odds(1-p)
    """
    p = max(epsilon, min(1.0 - epsilon, p))
    return math.log(p / (1.0 - p))


def log_odds_to_prob(log_odds: float) -> float:
    """
    Convert log-odds to probability using the logistic function.
    Handles overflow for large negative/positive log-odds.
    
    Formula: p = 1 / (1 + exp(-log_odds))
    """
    try:
        return 1.0 / (1.0 + math.exp(-log_odds))
    except OverflowError:
        return 0.0 if log_odds < 0 else 1.0


def update_belief(current_belief: BeliefState, evidence_log_odds: float) -> BeliefState:
    """
    Updates the belief state with new evidence using Bayesian inference.
    In log-odds space, Bayes' theorem becomes additive:
        Posterior(LO) = Prior(LO) + Likelihood_Ratio(LO)
    """
    new_log_odds = current_belief.log_odds + evidence_log_odds
    return BeliefState(log_odds=new_log_odds, confidence=current_belief.confidence)


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


def calculate_evidence_impact(verdict: str, confidence: float = 1.0, relevance: float = 1.0) -> float:
    """
    Calculates the log-odds impact of a single piece of evidence.
    Uses sigmoid saturation to dampen weak evidence.
    """
    v = verdict.lower()
    if v in ("verified", "true", "supported", "mostly true"):
        direction = 1.0
    elif v in ("refuted", "false", "pants on fire", "mostly false"):
        direction = -1.0
    elif v in ("mixed", "half true"):
        direction = 0.0
    else:
        direction = 0.0

    return sigmoid_impact(strength=confidence, relevance=relevance, direction=direction)


def process_updates(initial_belief: BeliefState, updates: List[float]) -> BeliefState:
    """
    Sequentially applies a list of log-odds updates to the belief state.
    """
    current = initial_belief
    for u in updates:
        current = update_belief(current, u)
    return current


def sigmoid_impact(
    strength: float, 
    relevance: float, 
    direction: float,
    k: float = 10.0, 
    x0: float = 0.5, 
    l_max: float = 2.0
) -> float:
    """
    Calculates non-linear impact using a sigmoid function.
    Weak or low-relevance claims saturate and have minimal impact.
    Strong claims approach L_max.
    
    Args:
        strength: Evidence strength/confidence [0, 1].
        relevance: Semantic relevance [0, 1].
        direction: +1.0 (Support), -1.0 (Refute).
        k: Steepness of sigmoid (default 10.0).
        x0: Midpoint (default 0.5).
        l_max: Maximum log-odds impact (default 2.0).
        
    Returns:
        Log-odds update value.
    """
    try:
        sigmoid_val = 1.0 / (1.0 + math.exp(-k * (strength - x0)))
    except OverflowError:
        sigmoid_val = 0.0 if (strength - x0) < 0 else 1.0

    impact = relevance * l_max * sigmoid_val
    return direction * impact