import math
from dataclasses import dataclass, field
from typing import List, Optional
from spectrue_core.schema.scoring import BeliefState, ConsensusState

def prob_to_log_odds(p: float, epsilon: float = 1e-9) -> float:
    """
    Convert probability to log-odds (logit function).
    Clips probability to [epsilon, 1-epsilon] to avoid infinity.
    
    Mathematical rationale:
    -----------------------
    Log-odds (also called "logit") is defined as:
        log_odds = log(p / (1 - p))
    
    This transformation is central to Bayesian inference because:
    
    1. **Additivity**: In log-odds space, Bayes' theorem becomes additive:
       Posterior(LO) = Prior(LO) + log(Likelihood Ratio)
       
       This makes sequential updates simple summations rather than
       complex probability products.
    
    2. **Unbounded range**: While probabilities are bounded [0, 1],
       log-odds range from -∞ to +∞, which is mathematically convenient
       for accumulating evidence.
    
    3. **Symmetry**: log_odds(p) = -log_odds(1-p)
       This means equal evidence for/against shifts belief symmetrically.
    
    Reference: Good, I.J. (1950). Probability and the Weighing of Evidence.
    """
    # Clipping prevents log(0) = -∞ or log(∞)
    p = max(epsilon, min(1.0 - epsilon, p))
    return math.log(p / (1.0 - p))

def log_odds_to_prob(log_odds: float) -> float:
    """
    Convert log-odds to probability using the logistic function.
    Handles overflow for large negative/positive log-odds.
    """
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
    # Current scope doesn't specify confidence updates, so we preserve it or it could be updated based on evidence variance.
    # For now, we return a new state with updated log-odds.
    return BeliefState(log_odds=new_log_odds, confidence=current_belief.confidence)

def apply_consensus_bound(belief: BeliefState, consensus: "ConsensusState") -> BeliefState:
    """
    Bounds the posterior belief using the Scientific Consensus latent variable.
    Effectively, the credibility of an article cannot exceed the scientific consensus 
    on the topics it covers.
    """
    # If insufficient sources for consensus, we don't apply a tight bound
    if consensus.source_count < 2:
        return belief
        
    # Calculate log-odds limit from consensus score
    # We treat consensus score as a probability cap
    limit_log_odds = prob_to_log_odds(consensus.score)
    
    new_log_odds = min(belief.log_odds, limit_log_odds)
    
    return BeliefState(log_odds=new_log_odds, confidence=belief.confidence)

def calculate_evidence_impact(verdict: str, confidence: float = 1.0, relevance: float = 1.0) -> float:
    """
    Calculates the log-odds impact of a single piece of evidence (claim verdict).
    Uses sigmoid saturation to dampen weak evidence.
    """
    v = verdict.lower()
    if v in ("verified", "true", "supported", "mostly true"):
        direction = 1.0
    elif v in ("refuted", "false", "pants on fire", "mostly false"):
        direction = -1.0
    elif v in ("mixed", "half true"):
        direction = 0.0 # Or maybe small penalty?
    else:
        direction = 0.0
        
    return sigmoid_impact(strength=confidence, relevance=relevance, direction=direction)

def process_updates(initial_belief: BeliefState, updates: List[float]) -> BeliefState:
    """
    Sequentially applies a list of log-odds updates to the belief state.
    """
    current = initial_belief
    for u in updates:
        current = update_belief(current, u)
    return current


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
        k: Steepness of sigmoid (default 10.0 for sharp transition).
        x0: Midpoint (default 0.5).
        l_max: Maximum log-odds impact (default 2.0).
        
    Returns:
        Log-odds update value.
    """
    # Sigmoid function
    # 1 / (1 + e^(-k(x - x0)))
    try:
        sigmoid_val = 1.0 / (1.0 + math.exp(-k * (strength - x0)))
    except OverflowError:
        sigmoid_val = 0.0 if (strength - x0) < 0 else 1.0
    
    # Scale by relevance and max impact
    impact = relevance * l_max * sigmoid_val
    
    return direction * impact


# =============================================================================
# FR-007: RGBA as Independent Probabilistic Belief Dimensions
# =============================================================================

@dataclass
class RGBABeliefState:
    """
    M104/FR-007: Independent Bayesian belief dimensions for RGBA.
    
    Each dimension (R, G, B, A) has its own:
    - Prior belief (set from source tier, context)
    - Evidence updates
    - Posterior belief
    
    Dimensions:
    - R (Red/Danger): Probability that content is harmful if believed
    - G (Green/Veracity): Probability that content is factually true
    - B (Blue/Honesty): Probability that content is presented in good faith
    - A (Alpha/Explainability): Probability that we can explain why we believe G
    
    Key principle: Dimensions are INDEPENDENT. Evidence for one doesn't
    automatically affect others (e.g., true but misleading content has
    high G but low B).
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
            danger_prior: P(harmful | source tier), default 0.5 (neutral)
            veracity_prior: P(true | source tier), e.g., 0.8 for Tier A
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


def create_rgba_belief_from_tier(tier: str) -> RGBABeliefState:
    """
    Factory function to create RGBABeliefState with tier-based priors.
    
    Maps evidence tiers to veracity priors:
    - Tier A (Official): 0.85 prior
    - Tier A' (Official Social): 0.75 prior
    - Tier B (Trusted Media): 0.70 prior
    - Tier C (Local Media): 0.55 prior
    - Tier D (Social): 0.35 prior
    """
    tier_priors = {
        "A": 0.85,
        "A'": 0.75,
        "B": 0.70,
        "C": 0.55,
        "D": 0.35,
    }
    
    veracity_prior = tier_priors.get(tier.upper(), 0.5)
    
    # Danger prior inversely related to tier (high-tier sources less likely harmful)
    danger_inverse = {
        "A": 0.1,
        "A'": 0.15,
        "B": 0.2,
        "C": 0.35,
        "D": 0.5,
    }
    danger_prior = danger_inverse.get(tier.upper(), 0.5)
    
    return RGBABeliefState.from_priors(
        danger_prior=danger_prior,
        veracity_prior=veracity_prior,
        honesty_prior=0.5,  # Start neutral
        explainability_prior=0.5,  # Start neutral
    )