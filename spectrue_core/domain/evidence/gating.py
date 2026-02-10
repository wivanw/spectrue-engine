"""EVOI gating logic for expensive evidence steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from .weights import StanceWeights, ClusterWeights, DEFAULT_STANCE_WEIGHTS, DEFAULT_CLUSTER_WEIGHTS
from .features import extract_evidence_features


# =============================================================================
# LOGISTIC MODEL WEIGHTS (calibratable priors)
# =============================================================================



@dataclass(frozen=True)
class GateDecisionPayload:
    enabled: bool
    p_need: float
    expected_gain: float
    expected_cost: float
    threshold: float
    reasons: tuple[str, ...]


# =============================================================================
# BAYESIAN UNCERTAINTY (BETA-BERNOULLI MODEL)
# =============================================================================
#
# We estimate: θ = "probability that evidence is sufficiently informative"
#
# Using Beta-Bernoulli with continuous noisy evidence:
#   - Each EvidenceItem contributes strength s_i ∈ (0,1)
#   - s_i = w_provider * provider_score + w_sim * sim
#   - Posterior: Beta(alpha, beta) where:
#     - alpha = alpha0 + Σ s_i
#     - beta = beta0 + Σ (1 - s_i)
#
# Uncertainty = normalized variance of posterior

# Evidence combination weights (policy priors, calibratable)
W_PROVIDER = 0.6  # Weight for provider_score
W_SIM = 0.4  # Weight for similarity score

# Prior hyperparameters (uninformative)
ALPHA0 = 1.0
BETA0 = 1.0

# Maximum variance for Beta(1,1)
VAR_MAX = 1.0 / 12.0


def _compute_evidence_strength(item: Any) -> float:
    """Compute evidence strength s_i ∈ (0.01, 0.99) from item scores."""
    provider = getattr(item, "provider_score", None)
    sim = getattr(item, "sim", None)
    provider = provider if provider is not None else 0.5
    sim = sim if sim is not None else 0.5

    s = W_PROVIDER * provider + W_SIM * sim
    return max(0.01, min(0.99, float(s)))


def _compute_update_weight(item: Any) -> float:
    """How much to trust this item's strength when updating the posterior."""
    has_provider = getattr(item, "provider_score", None) is not None
    has_sim = getattr(item, "sim", None) is not None
    if has_provider and has_sim:
        return 1.0
    if has_provider or has_sim:
        return 0.6
    return 0.2


def _collect_all_items(evidence_index: Any) -> list[Any]:
    """Collect evidence items from index (by_claim + global), de-duplicated by URL."""
    items: list[Any] = []
    seen: set[str] = set()

    def _push(it: Any) -> None:
        url = (getattr(it, "url", None) or "").strip()
        key = url or f"__no_url__:{id(it)}"
        if key in seen:
            return
        seen.add(key)
        items.append(it)

    for pack in getattr(evidence_index, "by_claim_id", {}).values():
        for it in getattr(pack, "items", ()) or ():
            _push(it)

    global_pack = getattr(evidence_index, "global_pack", None)
    if global_pack:
        for it in getattr(global_pack, "items", ()) or ():
            _push(it)

    return items


def compute_beta_uncertainty(evidence_index: Any) -> float:
    """Compute uncertainty as normalized Beta posterior variance."""
    items = _collect_all_items(evidence_index)

    if not items:
        return 1.0  # Maximum uncertainty with no evidence

    alpha = ALPHA0
    beta = BETA0

    for item in items:
        s = _compute_evidence_strength(item)
        w = _compute_update_weight(item)
        alpha += w * s
        beta += w * (1.0 - s)

    # Beta posterior variance
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    # Normalize to [0, 1]
    return var / VAR_MAX


def _compute_delta_utility(uncertainty: float, base_utility: float = 0.05) -> float:
    """Compute expected utility improvement from running a step."""
    k = 0.15
    return base_utility + k * uncertainty


# =============================================================================
# FEATURE EXTRACTION FROM EVIDENCE INDEX
# =============================================================================




# =============================================================================
# COST ESTIMATION
# =============================================================================


def get_ledger_history(ledger: Any, key: str) -> list[float]:
    """Safely extract cost history from ledger."""
    if ledger is None:
        return []

    if hasattr(ledger, "get_history"):
        return ledger.get_history(key) or []
    if isinstance(ledger, dict):
        return ledger.get(key, [])

    return []


def _estimate_cost_ema(history: list[float], fallback: float) -> float:
    """Compute EMA of cost history, or fallback."""
    if not history:
        return fallback

    alpha = 0.3
    ema = history[-1]
    for cost in reversed(history[-5:-1]):
        ema = alpha * cost + (1 - alpha) * ema
    return ema


def estimate_stance_cost(features: dict[str, float], ledger: Any) -> float:
    """Estimate stance cost from history or linear model."""
    history = get_ledger_history(ledger, "stance_costs")
    if history:
        return _estimate_cost_ema(history, 0.3)

    # Linear model fallback
    n_evidence = features["n_evidence"]
    n_claims = features["n_claims"]
    return 0.15 + 0.03 * n_evidence + 0.05 * n_claims


def estimate_cluster_cost(features: dict[str, float], ledger: Any) -> float:
    """Estimate cluster cost from history or linear model."""
    history = get_ledger_history(ledger, "cluster_costs")
    if history:
        return _estimate_cost_ema(history, 0.2)

    n_evidence = features["n_evidence"]
    n_claims = features["n_claims"]
    return 0.10 + 0.02 * n_evidence + 0.03 * n_claims


# =============================================================================
# SIGMOID MODEL
# =============================================================================


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
    delta_utility = _compute_delta_utility(uncertainty)
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
