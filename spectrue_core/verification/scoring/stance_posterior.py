# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Bayesian Stance Posterior Estimator (M113+)

Replaces hard rule-based stance classification with soft probabilistic posteriors.

Key idea: LLM stance is a NOISY OBSERVATION, not ground truth.
We model P(S* = k | features) where:
- S* = true stance
- features = {llm_stance, relevance, quote_present, has_chunk, source_prior}

This uses MAP (Maximum A Posteriori) with hand-tuned conservative priors.
No training data required initially - priors can be calibrated later.

Architecture:
- Multinomial logistic: P(S=k | x) = softmax(w_k · x + b_k)
- Hand-tuned weights encode domain knowledge
- Regularization via conservative biases (prior against strong stances without evidence)
"""

import math
from dataclasses import dataclass
from typing import Optional

# Stance classes
STANCE_SUPPORT = "support"
STANCE_REFUTE = "refute"
STANCE_NEUTRAL = "neutral"
STANCE_CONTEXT = "context"
STANCE_IRRELEVANT = "irrelevant"

STANCE_CLASSES = [STANCE_SUPPORT, STANCE_REFUTE, STANCE_NEUTRAL, STANCE_CONTEXT, STANCE_IRRELEVANT]


@dataclass
class StanceFeatures:
    """Structural features for stance posterior calculation."""

    # LLM observations (noisy)
    llm_stance: str  # Raw LLM prediction
    llm_relevance: Optional[float]  # 0-1 or None if missing

    # Structural signals (deterministic)
    quote_present: bool
    has_evidence_chunk: bool
    source_prior: float  # 0-1, derived from tier/domain quality

    # Optional: retrieval signals
    retrieval_rank: int = 0  # Position in search results (0 = first)
    retrieval_score: Optional[float] = None  # Search API score if available


@dataclass 
class StancePosterior:
    """Posterior probability distribution over stance classes."""

    p_support: float
    p_refute: float
    p_neutral: float
    p_context: float
    p_irrelevant: float

    # Derived metrics
    p_evidence: float  # P(S ∈ {SUPPORT, REFUTE})
    argmax_stance: str  # Most likely stance
    entropy: float  # Uncertainty measure

    def to_dict(self) -> dict:
        return {
            "p_support": round(self.p_support, 4),
            "p_refute": round(self.p_refute, 4),
            "p_neutral": round(self.p_neutral, 4),
            "p_context": round(self.p_context, 4),
            "p_irrelevant": round(self.p_irrelevant, 4),
            "p_evidence": round(self.p_evidence, 4),
            "argmax_stance": self.argmax_stance,
            "entropy": round(self.entropy, 4),
        }


# ============================================================================
# HAND-TUNED WEIGHTS (Conservative Priors)
# ============================================================================
# These encode domain knowledge. Can be calibrated with gold data later.
# 
# Design principles:
# 1. LLM stance is a strong but fallible signal (~70% accuracy assumed)
# 2. Quote presence is a strong structural signal for S ∈ {SUPPORT, REFUTE}
# 3. Without evidence chunk, bias toward CONTEXT/IRRELEVANT
# 4. Relevance is a weak signal (LLM often miscalibrates)
# 5. Source prior (tier) provides mild regularization

WEIGHTS = {
    # Feature weights per stance class
    # Format: {stance: {feature: weight}}

    STANCE_SUPPORT: {
        "llm_said_support": 2.5,    # Strong signal but not deterministic
        "llm_said_refute": -1.5,    # Unlikely if LLM said refute
        "quote_present": 1.8,       # Quotes strongly indicate evidence
        "has_chunk": 0.8,           # Chunk is weaker signal
        "relevance": 1.2,           # Relevance helps but noisy
        "source_prior": 0.5,        # Tier quality mild boost
        "bias": -2.5,               # Prior against SUPPORT without evidence
    },

    STANCE_REFUTE: {
        "llm_said_support": -1.5,
        "llm_said_refute": 2.5,
        "quote_present": 1.8,
        "has_chunk": 0.8,
        "relevance": 1.2,
        "source_prior": 0.5,
        "bias": -3.0,               # Slightly stronger prior against REFUTE
    },

    STANCE_NEUTRAL: {
        "llm_said_support": 0.3,    # Weak positive (LLM might be right)
        "llm_said_refute": 0.3,
        "quote_present": 0.5,       # Some evidence, but not decisively for/against
        "has_chunk": 0.3,
        "relevance": 0.8,           # Relevant but no stance
        "source_prior": 0.2,
        "bias": -0.5,               # Mild prior below context
    },

    STANCE_CONTEXT: {
        "llm_said_support": -0.3,
        "llm_said_refute": -0.3,
        "quote_present": -0.5,      # Quotes suggest stronger stance
        "has_chunk": 0.2,
        "relevance": 0.5,
        "source_prior": 0.1,
        "bias": 0.0,                # Baseline stance
    },

    STANCE_IRRELEVANT: {
        "llm_said_support": -1.0,
        "llm_said_refute": -1.0,
        "quote_present": -1.5,
        "has_chunk": -1.0,
        "relevance": -2.0,          # Strong negative for relevance
        "source_prior": -0.3,
        "bias": 0.5,                # Mild prior toward irrelevant (safe default)
    },
}


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_logits)
    return [e / total for e in exp_logits]


def _entropy(probs: list[float]) -> float:
    """Shannon entropy of probability distribution."""
    return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)


def compute_stance_posterior(features: StanceFeatures) -> StancePosterior:
    """
    Compute posterior P(S* = k | features) using MAP with hand-tuned priors.
    
    This is the core Bayesian inference step that replaces hard rule-based logic.
    """

    # Normalize LLM stance
    llm_stance_lower = (features.llm_stance or "").lower()

    # Build feature vector
    feature_values = {
        "llm_said_support": 1.0 if llm_stance_lower in ("support", "sup") else 0.0,
        "llm_said_refute": 1.0 if llm_stance_lower in ("refute", "ref", "contradict") else 0.0,
        "quote_present": 1.0 if features.quote_present else 0.0,
        "has_chunk": 1.0 if features.has_evidence_chunk else 0.0,
        "relevance": features.llm_relevance if features.llm_relevance is not None else 0.5,  # Default 0.5 if missing
        "source_prior": features.source_prior,
    }

    # Compute logits for each stance
    logits = []
    for stance in STANCE_CLASSES:
        weights = WEIGHTS[stance]
        logit = weights["bias"]
        for feat_name, feat_val in feature_values.items():
            logit += weights.get(feat_name, 0.0) * feat_val
        logits.append(logit)

    # Softmax to get probabilities
    probs = _softmax(logits)

    # Build result
    p_support = probs[0]
    p_refute = probs[1]
    p_neutral = probs[2]
    p_context = probs[3]
    p_irrelevant = probs[4]

    p_evidence = p_support + p_refute

    # Argmax
    max_idx = probs.index(max(probs))
    argmax_stance = STANCE_CLASSES[max_idx]

    return StancePosterior(
        p_support=p_support,
        p_refute=p_refute,
        p_neutral=p_neutral,
        p_context=p_context,
        p_irrelevant=p_irrelevant,
        p_evidence=p_evidence,
        argmax_stance=argmax_stance,
        entropy=_entropy(probs),
    )


def compute_effective_evidence_count(posteriors: list[StancePosterior]) -> dict:
    """
    Compute expected evidence counts from posteriors.
    
    Replaces hard count(stance == SUPPORT) with E[evidence].
    """
    effective_support = sum(p.p_support for p in posteriors)
    effective_refute = sum(p.p_refute for p in posteriors)
    effective_neutral = sum(p.p_neutral for p in posteriors)
    effective_evidence = sum(p.p_evidence for p in posteriors)

    return {
        "effective_support": round(effective_support, 3),
        "effective_refute": round(effective_refute, 3),
        "effective_neutral": round(effective_neutral, 3),
        "effective_evidence": round(effective_evidence, 3),
        "total_sources": len(posteriors),
        "mean_entropy": round(sum(p.entropy for p in posteriors) / max(len(posteriors), 1), 3),
    }


def source_prior_from_tier(tier: Optional[str]) -> float:
    """
    Convert evidence tier to source prior probability.
    
    Tier A (official/primary) → 0.9
    Tier B (trusted news) → 0.7
    Tier C (general) → 0.5
    Tier D (low quality) → 0.3
    Unknown → 0.5
    """
    tier_map = {
        "A": 0.9,
        "B": 0.7,
        "C": 0.5,
        "D": 0.3,
    }
    if tier:
        return tier_map.get(tier.upper(), 0.5)
    return 0.5


# ============================================================================
# DEBUGGING / EXPLAINABILITY
# ============================================================================

def explain_posterior(features: StanceFeatures, posterior: StancePosterior) -> dict:
    """
    Generate human-readable explanation of posterior calculation.
    Useful for debugging and audit trails.
    """
    llm_stance_lower = (features.llm_stance or "").lower()

    contributions = {}
    for stance in STANCE_CLASSES:
        weights = WEIGHTS[stance]
        contrib = {"bias": weights["bias"]}

        feature_values = {
            "llm_said_support": 1.0 if llm_stance_lower in ("support", "sup") else 0.0,
            "llm_said_refute": 1.0 if llm_stance_lower in ("refute", "ref") else 0.0,
            "quote_present": 1.0 if features.quote_present else 0.0,
            "has_chunk": 1.0 if features.has_evidence_chunk else 0.0,
            "relevance": features.llm_relevance if features.llm_relevance is not None else 0.5,
            "source_prior": features.source_prior,
        }

        for feat_name, feat_val in feature_values.items():
            weight = weights.get(feat_name, 0.0)
            contrib[feat_name] = round(weight * feat_val, 3)

        contrib["total_logit"] = sum(contrib.values())
        contributions[stance] = contrib

    return {
        "features": {
            "llm_stance": features.llm_stance,
            "llm_relevance": features.llm_relevance,
            "quote_present": features.quote_present,
            "has_evidence_chunk": features.has_evidence_chunk,
            "source_prior": features.source_prior,
        },
        "logit_contributions": contributions,
        "posterior": posterior.to_dict(),
    }
