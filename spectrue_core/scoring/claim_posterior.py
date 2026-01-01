"""
Unified Bayesian Claim Posterior Model.

Single source of truth for claim veracity scoring. Replaces the double-counting
pattern where evidence was factored into both aggregate_claim_verdict() and
_compute_article_g_from_anchor().

Mathematical Foundation:
-----------------------
All signals are combined in log-odds space:

    ℓ_post = ℓ_prior + α·ℓ_llm + β·ℓ_evidence
    p_post = σ(ℓ_post)

Where:
    ℓ_prior = logit(p_prior)  -- from tier/domain quality
    ℓ_llm = logit(p_llm)      -- from LLM verdict_score
    ℓ_evidence = Σ w_j · s_j  -- weighted sum of stance signals
    
    s_j ∈ {+1, -1, 0}  -- stance direction (support/refute/neutral)
    w_j = stance_weight(tier, relevance, quote_present)

Parameters α, β come from policy profile (not hardcoded).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace


@dataclass(frozen=True)
class PosteriorParams:
    """
    Calibration parameters for the posterior model.
    
    These should come from policy_profile, not be hardcoded.
    Default values are conservative (equal weight to all signals).
    """
    alpha: float = 1.0  # Weight for LLM signal
    beta: float = 1.0   # Weight for evidence signal
    prior_default: float = 0.5  # Default prior when tier unknown

    # Tier -> prior probability mapping
    tier_priors: dict[str, float] = field(default_factory=lambda: {
        "A": 0.7,   # High-quality source shifts prior toward trust
        "A'": 0.65,
        "B": 0.6,
        "C": 0.5,   # Neutral
        "D": 0.4,   # Low-quality shifts prior toward skepticism
    })

    # Stance weights for evidence aggregation
    stance_weights: dict[str, float] = field(default_factory=lambda: {
        "quote_present": 1.5,    # Strong structural signal
        "high_relevance": 1.2,   # relevance > 0.7
        "medium_relevance": 0.8, # 0.4 < relevance <= 0.7
        "low_relevance": 0.3,    # relevance <= 0.4
    })


@dataclass
class EvidenceItem:
    """Single piece of evidence for posterior computation."""
    stance: str  # "support", "refute", "neutral", "context"
    tier: str | None
    relevance: float
    quote_present: bool
    claim_id: str | None = None


@dataclass
class PosteriorResult:
    """Result of posterior computation with full debug info."""
    p_posterior: float
    log_odds_prior: float
    log_odds_llm: float
    log_odds_evidence: float
    log_odds_posterior: float

    # Breakdown
    n_support: int
    n_refute: int
    n_neutral: int
    effective_support: float
    effective_refute: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_posterior": round(self.p_posterior, 4),
            "log_odds_prior": round(self.log_odds_prior, 4),
            "log_odds_llm": round(self.log_odds_llm, 4),
            "log_odds_evidence": round(self.log_odds_evidence, 4),
            "log_odds_posterior": round(self.log_odds_posterior, 4),
            "n_support": self.n_support,
            "n_refute": self.n_refute,
            "n_neutral": self.n_neutral,
            "effective_support": round(self.effective_support, 4),
            "effective_refute": round(self.effective_refute, 4),
        }


def _logit(p: float, eps: float = 1e-9) -> float:
    """Convert probability to log-odds. Clips to avoid infinity."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Convert log-odds to probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _stance_direction(stance: str) -> float:
    """Map stance to direction: +1 (support), -1 (refute), 0 (neutral/context)."""
    s = (stance or "").lower().strip()
    if s in ("support", "sup", "supported"):
        return 1.0
    if s in ("refute", "ref", "refuted", "contradict"):
        return -1.0
    return 0.0


def _evidence_weight(
    item: EvidenceItem,
    params: PosteriorParams,
) -> float:
    """
    Compute weight for a single evidence item.
    
    Weight = base_weight * relevance_factor * quote_factor
    """
    # Base weight from relevance
    rel = item.relevance if item.relevance is not None else 0.5
    if rel > 0.7:
        base = params.stance_weights.get("high_relevance", 1.2)
    elif rel > 0.4:
        base = params.stance_weights.get("medium_relevance", 0.8)
    else:
        base = params.stance_weights.get("low_relevance", 0.3)

    # Quote boost
    if item.quote_present:
        base *= params.stance_weights.get("quote_present", 1.5)

    return base


def compute_claim_posterior(
    *,
    llm_verdict_score: float,
    best_tier: str | None,
    evidence_items: list[EvidenceItem],
    params: PosteriorParams | None = None,
    claim_id: str | None = None,
) -> PosteriorResult:
    """
    Compute unified claim posterior from all signals.
    
    This is the SINGLE place where prior, LLM, and evidence are combined.
    No double-counting.
    
    Args:
        llm_verdict_score: Raw LLM prediction (0-1)
        best_tier: Best evidence tier for this claim (A/B/C/D)
        evidence_items: List of evidence items with stance/tier/relevance
        params: Calibration parameters (from policy profile)
        claim_id: For tracing
        
    Returns:
        PosteriorResult with p_posterior and debug info
    """
    if params is None:
        params = PosteriorParams()

    # 1. Prior from tier
    p_prior = params.tier_priors.get(
        (best_tier or "").upper(),
        params.prior_default
    )
    l_prior = _logit(p_prior)

    # 2. LLM signal
    p_llm = max(0.01, min(0.99, llm_verdict_score))
    l_llm = _logit(p_llm)

    # 3. Evidence signal: Σ w_j · s_j
    l_evidence = 0.0
    n_support = 0
    n_refute = 0
    n_neutral = 0
    effective_support = 0.0
    effective_refute = 0.0

    for item in evidence_items:
        direction = _stance_direction(item.stance)
        weight = _evidence_weight(item, params)

        if direction > 0:
            n_support += 1
            effective_support += weight
        elif direction < 0:
            n_refute += 1
            effective_refute += weight
        else:
            n_neutral += 1

        l_evidence += direction * weight

    # 4. Combine in log-odds space
    l_posterior = l_prior + params.alpha * l_llm + params.beta * l_evidence
    p_posterior = _sigmoid(l_posterior)

    result = PosteriorResult(
        p_posterior=p_posterior,
        log_odds_prior=l_prior,
        log_odds_llm=l_llm,
        log_odds_evidence=l_evidence,
        log_odds_posterior=l_posterior,
        n_support=n_support,
        n_refute=n_refute,
        n_neutral=n_neutral,
        effective_support=effective_support,
        effective_refute=effective_refute,
    )

    Trace.event(
        "claim.posterior.computed",
        {
            "claim_id": claim_id,
            "p_prior": round(p_prior, 4),
            "p_llm": round(p_llm, 4),
            "best_tier": best_tier,
            "alpha": params.alpha,
            "beta": params.beta,
            **result.to_dict(),
        },
    )

    return result


def aggregate_article_posterior(
    claim_posteriors: list[tuple[str, float]],
    method: str = "max",
) -> float:
    """
    Aggregate claim posteriors to article-level G.
    
    Args:
        claim_posteriors: List of (claim_id, p_posterior) tuples
        method: "max" (most extreme) or "mean" (weighted average)
        
    Returns:
        Article-level G score
    """
    if not claim_posteriors:
        return 0.5  # Neutral

    posteriors = [p for _, p in claim_posteriors]

    if method == "max":
        # Take the most extreme posterior (furthest from 0.5)
        return max(posteriors, key=lambda p: abs(p - 0.5))
    elif method == "mean":
        return sum(posteriors) / len(posteriors)
    else:
        return max(posteriors, key=lambda p: abs(p - 0.5))

