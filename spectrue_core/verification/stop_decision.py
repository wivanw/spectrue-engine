from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.pipeline_builder.spec import PipelineProfile


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


# ─────────────────────────────────────────────────────────────────────────────
# M113: Pipeline Profile Integration
# ─────────────────────────────────────────────────────────────────────────────


def ev_stop_params_from_pipeline_profile(
    profile: "PipelineProfile",
) -> EVStopParams | None:
    """
    Create EVStopParams from a pipeline profile's stop_policy configuration.

    Args:
        profile: Pipeline profile with stop_policy settings

    Returns:
        EVStopParams if stop policy is enabled, None otherwise
    """

    stop_policy = getattr(profile, "stop_policy", None)
    if stop_policy is None or not getattr(stop_policy, "enabled", False):
        return None

    return EVStopParams(
        value_uncertainty=getattr(stop_policy, "value_uncertainty", 1.0),
        marginal_cost=getattr(stop_policy, "marginal_cost", 0.25),
        min_entropy=getattr(stop_policy, "min_entropy", 0.15),
    )


@dataclass
class StopDecisionResult:
    """
    Result of a stop decision evaluation.

    Used for structured trace output.
    """
    should_stop: bool
    reason: str
    entropy: float
    expected_gain: float | None = None
    marginal_cost: float | None = None
    budget_remaining: float | None = None
    quality_signal: dict | None = None

    def to_dict(self) -> dict:
        """Serialize for trace output."""
        result = {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "entropy": self.entropy,
        }
        if self.expected_gain is not None:
            result["expected_gain"] = self.expected_gain
        if self.marginal_cost is not None:
            result["marginal_cost"] = self.marginal_cost
        if self.budget_remaining is not None:
            result["budget_remaining"] = self.budget_remaining
        if self.quality_signal:
            result["quality_signal"] = self.quality_signal
        return result


def evaluate_stop_decision(
    *,
    posterior_true: float,
    expected_delta_p: Optional[float],
    params: EVStopParams | None = None,
    budget_remaining: float | None = None,
    quality_signal: dict | None = None,
) -> StopDecisionResult:
    """
    Evaluate stop decision with full trace output.

    This is the M113 version of should_stop_by_ev with structured output
    suitable for trace recording.

    Args:
        posterior_true: Current posterior probability
        expected_delta_p: Expected change in posterior from another hop
        params: Stop parameters (from profile or defaults)
        budget_remaining: Remaining budget credits
        quality_signal: Quality metrics from retrieval evaluation

    Returns:
        StopDecisionResult with full decision details
    """
    should_stop, details = should_stop_by_ev(
        posterior_true=posterior_true,
        expected_delta_p=expected_delta_p,
        params=params,
    )

    return StopDecisionResult(
        should_stop=should_stop,
        reason=details.get("stop_reason", "unknown"),
        entropy=details.get("entropy", 0.0),
        expected_gain=details.get("expected_gain"),
        marginal_cost=details.get("marginal_cost"),
        budget_remaining=budget_remaining,
        quality_signal=quality_signal,
    )

