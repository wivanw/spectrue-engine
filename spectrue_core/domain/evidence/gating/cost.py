"""Cost estimation logic from ledger history."""
from typing import Any

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
