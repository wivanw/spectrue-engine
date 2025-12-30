from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EVStopParams:
    """
    Deterministic Expected-Value stop model.

    We stop when the expected uncertainty reduction from one more search hop
    is lower than the marginal cost of that hop.
    """

    # How valuable uncertainty reduction is (domain-level constant)
    value_uncertainty: float = 1.0

    # Marginal cost of one additional hop (search + extract)
    marginal_cost: float = 0.25

    # Minimum entropy below which further search is pointless
    min_entropy: float = 0.15


def entropy_bernoulli(p: float) -> float:
    """
    Shannon entropy of Bernoulli(p), normalized to [0,1].
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    from math import log2

    h = -p * log2(p) - (1.0 - p) * log2(1.0 - p)
    return h


def should_stop_by_ev(
    *,
    posterior_true: float,
    expected_delta_p: Optional[float],
    params: EVStopParams | None = None,
) -> tuple[bool, dict]:
    """
    Decide whether to stop further search based on expected value.

    expected_delta_p:
        Expected absolute change in posterior from one more hop.
        Caller provides this (from calibration / retrieval_gain).
    """
    p = max(1e-6, min(1.0 - 1e-6, float(posterior_true)))
    params = params or EVStopParams()

    entropy = entropy_bernoulli(p)

    # If already very certain, stop.
    if entropy <= params.min_entropy:
        return True, {
            "stop_reason": "low_entropy",
            "entropy": entropy,
        }

    if not expected_delta_p or expected_delta_p <= 0.0:
        return True, {
            "stop_reason": "no_expected_gain",
            "entropy": entropy,
        }

    # Expected uncertainty reduction ≈ |Δp| * entropy
    expected_gain = params.value_uncertainty * abs(expected_delta_p) * entropy

    if expected_gain < params.marginal_cost:
        return True, {
            "stop_reason": "ev_negative",
            "entropy": entropy,
            "expected_gain": expected_gain,
            "marginal_cost": params.marginal_cost,
        }

    return False, {
        "stop_reason": "continue",
        "entropy": entropy,
        "expected_gain": expected_gain,
    }
