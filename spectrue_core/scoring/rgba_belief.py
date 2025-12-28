"""
M104/FR-007: RGBA as Independent Probabilistic Belief Dimensions.

This module provides RGBABeliefState - a dataclass representing independent
Bayesian beliefs for each RGBA dimension (Danger, Veracity, Honesty, Explainability).

Key principle: Dimensions are INDEPENDENT. Evidence for one doesn't
automatically affect others (e.g., true but misleading content has
high Veracity but low Honesty).
"""

from dataclasses import dataclass, field

from spectrue_core.schema.scoring import BeliefState
from spectrue_core.scoring.belief import prob_to_log_odds


@dataclass
class RGBABeliefState:
    """
    Independent Bayesian belief dimensions for RGBA.
    
    Each dimension (R, G, B, A) has its own:
    - Prior belief (set from context/initialization)
    - Evidence updates
    - Posterior belief
    
    Dimensions:
    - R (Red/Danger): Probability that content is harmful if believed
    - G (Green/Veracity): Probability that content is factually true
    - B (Blue/Honesty): Probability that content is presented in good faith
    - A (Alpha/Explainability): Probability that we can explain why we believe G
    """
    
    danger: BeliefState = field(default_factory=lambda: BeliefState(log_odds=0.0, confidence=0.5))
    veracity: BeliefState = field(default_factory=lambda: BeliefState(log_odds=0.0, confidence=0.5))
    honesty: BeliefState = field(default_factory=lambda: BeliefState(log_odds=0.0, confidence=0.5))
    explainability: BeliefState = field(default_factory=lambda: BeliefState(log_odds=0.0, confidence=0.5))
    
    @classmethod
    def from_priors(
        cls,
        *,
        danger_prior: float = 0.5,
        veracity_prior: float = 0.5,
        honesty_prior: float = 0.5,
        explainability_prior: float = 0.5,
    ) -> "RGBABeliefState":
        """
        Initialize RGBA beliefs from probability priors.
        
        Args:
            danger_prior: P(harmful), default 0.5 (neutral)
            veracity_prior: P(true), default 0.5 (neutral)
            honesty_prior: P(good faith), default 0.5
            explainability_prior: P(explainable), default 0.5
        """
        return cls(
            danger=BeliefState(log_odds=prob_to_log_odds(danger_prior), confidence=0.5),
            veracity=BeliefState(log_odds=prob_to_log_odds(veracity_prior), confidence=0.5),
            honesty=BeliefState(log_odds=prob_to_log_odds(honesty_prior), confidence=0.5),
            explainability=BeliefState(log_odds=prob_to_log_odds(explainability_prior), confidence=0.5),
        )
    
    def update_veracity(self, evidence_log_odds: float, confidence_delta: float = 0.0) -> None:
        """Update veracity belief with new evidence (in-place)."""
        new_log_odds = self.veracity.log_odds + evidence_log_odds
        new_confidence = min(1.0, self.veracity.confidence + confidence_delta)
        self.veracity = BeliefState(log_odds=new_log_odds, confidence=new_confidence)
    
    def update_danger(self, evidence_log_odds: float, confidence_delta: float = 0.0) -> None:
        """Update danger belief with new evidence (in-place)."""
        new_log_odds = self.danger.log_odds + evidence_log_odds
        new_confidence = min(1.0, self.danger.confidence + confidence_delta)
        self.danger = BeliefState(log_odds=new_log_odds, confidence=new_confidence)
    
    def update_honesty(self, evidence_log_odds: float, confidence_delta: float = 0.0) -> None:
        """Update honesty belief with new evidence (in-place)."""
        new_log_odds = self.honesty.log_odds + evidence_log_odds
        new_confidence = min(1.0, self.honesty.confidence + confidence_delta)
        self.honesty = BeliefState(log_odds=new_log_odds, confidence=new_confidence)
    
    def update_explainability(self, evidence_log_odds: float, confidence_delta: float = 0.0) -> None:
        """Update explainability belief with new evidence (in-place)."""
        new_log_odds = self.explainability.log_odds + evidence_log_odds
        new_confidence = min(1.0, self.explainability.confidence + confidence_delta)
        self.explainability = BeliefState(log_odds=new_log_odds, confidence=new_confidence)
    
    def to_probabilities(self) -> tuple[float, float, float, float]:
        """
        Convert beliefs to probability array [R, G, B, A].
        Compatible with legacy RGBA format.
        """
        return (
            self.danger.probability,
            self.veracity.probability,
            self.honesty.probability,
            self.explainability.probability,
        )
    
    def to_dict(self) -> dict:
        """Serialize for API response."""
        r, g, b, a = self.to_probabilities()
        return {
            "rgba": [round(r, 3), round(g, 3), round(b, 3), round(a, 3)],
            "belief_state": {
                "danger": {"log_odds": round(self.danger.log_odds, 3), "confidence": round(self.danger.confidence, 3)},
                "veracity": {"log_odds": round(self.veracity.log_odds, 3), "confidence": round(self.veracity.confidence, 3)},
                "honesty": {"log_odds": round(self.honesty.log_odds, 3), "confidence": round(self.honesty.confidence, 3)},
                "explainability": {"log_odds": round(self.explainability.log_odds, 3), "confidence": round(self.explainability.confidence, 3)},
            },
        }


_TIER_EXPLAINABILITY_PRIORS = {
    # IMPORTANT: Tier is a prior factor for A (Explainability) ONLY.
    # It must not bias veracity (G) or verdicts.
    # Values here represent expected explainability (how likely we can trace/quote).
    "A": 0.96,
    "A'": 0.93,
    "B": 0.90,
    "C": 0.85,
    "D": 0.80,
    "UNKNOWN": 0.38,
}


def create_rgba_belief_from_tier(tier: str) -> RGBABeliefState:
    """
    Factory function to create RGBABeliefState with tier-based priors.
    
    Tier affects Alpha (Explainability) only.
    Veracity (G), Danger (R) and Honesty (B) start neutral (0.5) and are driven by evidence.
    """
    explainability_prior = _TIER_EXPLAINABILITY_PRIORS.get(
        str(tier).strip().upper(), _TIER_EXPLAINABILITY_PRIORS["UNKNOWN"]
    )
    
    return RGBABeliefState.from_priors(
        danger_prior=0.5,
        veracity_prior=0.5,
        honesty_prior=0.5,
        explainability_prior=explainability_prior,
    )
