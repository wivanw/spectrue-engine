from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BeliefState:
    log_odds: float
    confidence: float = 0.0

    @property
    def probability(self) -> float:
        return log_odds_to_prob(self.log_odds)


@dataclass
class ConsensusState:
    score: float
    stability: float
    source_count: int


def prob_to_log_odds(p: float, epsilon: float = 1e-9) -> float:
    """Convert probability to log-odds (logit function)."""
    p = max(epsilon, min(1.0 - epsilon, p))
    return math.log(p / (1.0 - p))


def log_odds_to_prob(log_odds: float) -> float:
    """Convert log-odds to probability (logistic function)."""
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


def process_updates(initial_belief: BeliefState, updates: list[float]) -> BeliefState:
    """
    Sequentially applies a list of log-odds updates to the belief state.
    """
    current = initial_belief
    for u in updates:
        current = update_belief(current, u)
    return current


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
