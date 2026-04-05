"""EVOI gating logic for expensive evidence steps."""

import math
from typing import Any

from spectrue_core.domain.evidence.weights import (
    StanceWeights,
    ClusterWeights,
    DEFAULT_STANCE_WEIGHTS,
    DEFAULT_CLUSTER_WEIGHTS,
)
from .models import GateDecisionPayload
from .cost import estimate_stance_cost, estimate_cluster_cost
from .uncertainty import compute_delta_utility


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_stance_gate(
    features: dict[str, float],
    uncertainty: float,
    ledger: Any,
    weights: StanceWeights = DEFAULT_STANCE_WEIGHTS,
    cost_weight: float = 1.0,
) -> GateDecisionPayload:
    """Compute stance gate using logistic model."""
    contributions = {
        "intercept": weights.intercept,
        "unlabeled": weights.unlabeled_ratio * features["unlabeled_ratio"],
        "low_tier": weights.low_tier_ratio * features["low_tier_ratio"],
        "log_claims": weights.log_claims * features["log_claims"],
        "uncertainty": weights.uncertainty * uncertainty,
    }

    logit = sum(contributions.values())
    p_need = _sigmoid(logit)

    expected_cost = estimate_stance_cost(features, ledger)
    delta_utility = compute_delta_utility(uncertainty)
    expected_gain = p_need * delta_utility
    threshold = expected_cost * cost_weight

    enabled = expected_gain > threshold

    reasons = [f"{k}:{v:.3f}" for k, v in contributions.items()]
    reasons.append(f"logit:{logit:.2f}")
    reasons.append(f"p_need:{p_need:.3f}")

    return GateDecisionPayload(
        enabled=enabled,
        p_need=p_need,
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


def compute_cluster_gate(
    features: dict[str, float],
    stance_enabled: bool,
    ledger: Any,
    weights: ClusterWeights = DEFAULT_CLUSTER_WEIGHTS,
    cost_weight: float = 1.0,
) -> GateDecisionPayload:
    """Compute cluster gate using logistic model."""
    if not stance_enabled:
        expected_cost = estimate_cluster_cost(features, ledger)
        return GateDecisionPayload(
            enabled=False,
            p_need=0.0,
            expected_gain=0.0,
            expected_cost=expected_cost,
            threshold=expected_cost * cost_weight,
            reasons=("stance_disabled",),
        )

    contributions = {
        "intercept": weights.intercept,
        "log_claims": weights.log_claims * features["log_claims"],
    }

    logit = sum(contributions.values())
    p_need = _sigmoid(logit)

    expected_cost = estimate_cluster_cost(features, ledger)
    delta_utility = 0.08
    expected_gain = p_need * delta_utility
    threshold = expected_cost * cost_weight

    enabled = expected_gain > threshold

    reasons = [f"{k}:{v:.3f}" for k, v in contributions.items()]
    reasons.append(f"logit:{logit:.2f}")

    return GateDecisionPayload(
        enabled=enabled,
        p_need=p_need,
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )
