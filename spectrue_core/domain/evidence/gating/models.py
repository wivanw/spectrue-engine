from dataclasses import dataclass

@dataclass(frozen=True)
class GateDecisionPayload:
    enabled: bool
    p_need: float
    expected_gain: float
    expected_cost: float
    threshold: float
    reasons: tuple[str, ...]
