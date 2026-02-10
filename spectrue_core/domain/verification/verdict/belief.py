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
