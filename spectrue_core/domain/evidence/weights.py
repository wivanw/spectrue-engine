from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StanceWeights:
    """Logistic regression weights for P(stance needed)."""

    intercept: float = -0.5
    unlabeled_ratio: float = 0.8
    low_tier_ratio: float = 0.5
    log_claims: float = 0.3
    uncertainty: float = 2.5  # Weight for evidence uncertainty


@dataclass(frozen=True)
class ClusterWeights:
    """Logistic regression weights for P(cluster needed)."""

    intercept: float = -2.0
    log_claims: float = 0.6


DEFAULT_STANCE_WEIGHTS = StanceWeights()
DEFAULT_CLUSTER_WEIGHTS = ClusterWeights()
