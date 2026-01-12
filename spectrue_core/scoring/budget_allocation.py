# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Bayesian Budget Allocation for Dynamic Extract and Query Limits.

This module uses the same EVOI (Expected Value of Information) framework
as target_selection.py to dynamically compute:
1. Maximum extracts based on evidence quality (marginal value vs cost)
2. Number of search queries based on claim complexity

Mathematical Foundation:
------------------------
Uses the same log-odds Bayesian model as claim_posterior.py:

    ℓ_post = ℓ_prior + α·ℓ_llm + β·ℓ_evidence

For budget allocation, we model:
    - P(useful_extract | current_evidence) via Beta-Bernoulli conjugate
    - EVOI(next_extract) = value_uncertainty × entropy × quality_factors
    - Stop when: EVOI(next) < marginal_cost

Reference: Same foundations as claim_posterior.py (log-odds, Bayesian updates)
See: docs/ALGORITHMS.md, docs/CALIBRATION.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace


# -----------------------------------------------------------------------------
# Mathematical Functions (consistent with scoring/belief.py)
# -----------------------------------------------------------------------------

def _logit(p: float, eps: float = 1e-9) -> float:
    """Convert probability to log-odds."""
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Convert log-odds to probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _entropy_bernoulli(p: float) -> float:
    """
    Shannon entropy of Bernoulli(p), normalized to [0,1].
    Max entropy at p=0.5 (maximum uncertainty).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return h


# -----------------------------------------------------------------------------
# Budget Parameters (consistent with TargetBudgetParams pattern)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractBudgetParams:
    """
    Parameters for Bayesian extract budget computation.
    
    Uses same EVOI framework as target_selection.py:
        Continue extracting while: EVOI(next) > marginal_cost
    
    The model naturally stops when:
    - Evidence sufficiency is high (low entropy → low EVOI)
    - Hit rate is low (low posterior_mean → low expected quality)
    - Diminishing returns kick in (decay factor)
    """
    # Cost per extract (Tavily extract ≈ 0.5 credits)
    marginal_cost_per_extract: float = 0.5
    
    # Base value of uncertainty reduction
    value_uncertainty: float = 1.0
    
    # Diminishing returns: EVOI_i = base × decay^i
    diminishing_returns_decay: float = 0.85
    
    # Minimum EVOI threshold to continue extracting
    min_evoi_threshold: float = 0.08
    
    # Beta prior parameters for hit rate estimation
    # Beta(alpha_prior, beta_prior) = initial belief about P(useful_extract)
    alpha_prior: float = 2.0  # Prior successes (optimistic)
    beta_prior: float = 2.0   # Prior failures
    
    # Hard limits (safety bounds)
    min_extracts: int = 2
    max_extracts: int = 18
    
    # Quality signal weights (for EVOI computation)
    # w_quote = extra weight when source has quote
    # w_tier = multiplier for authoritative sources
    weight_quote: float = 1.5
    weight_authoritative: float = 1.2
    weight_relevance: float = 1.0


@dataclass(frozen=True)
class QueryBudgetParams:
    """
    Parameters for Bayesian query count computation.
    
    Query count depends on:
    - Claim complexity (affects search difficulty)
    - Named entities (enable targeted queries)
    - Current evidence sufficiency
    """
    # Base query count
    base_queries: int = 3
    
    # Complexity thresholds
    high_complexity_threshold: float = 0.7
    low_complexity_threshold: float = 0.3
    
    # Hard limits
    min_queries: int = 1
    max_queries: int = 6


# -----------------------------------------------------------------------------
# Budget State (Bayesian tracking)
# -----------------------------------------------------------------------------

@dataclass
class BudgetState:
    """
    Tracks Bayesian state for extract budget decisions.
    
    Uses Beta-Bernoulli conjugate prior for hit rate estimation:
        P(useful) ~ Beta(alpha, beta)
    
    After observing success/failure:
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + failures
    
    Posterior mean = alpha / (alpha + beta)
    """
    # Beta distribution parameters (updated via conjugate)
    alpha: float = 2.0
    beta: float = 2.0
    
    # Observation counts
    extracts_used: int = 0
    quotes_found: int = 0
    relevant_sources: int = 0
    total_sources: int = 0
    
    # Quality metrics
    sum_relevance: float = 0.0
    max_relevance: float = 0.0
    authoritative_count: int = 0
    
    # Configuration
    params: ExtractBudgetParams = field(default_factory=ExtractBudgetParams)
    
    @property
    def posterior_mean(self) -> float:
        """Expected P(next extract is useful) from Beta posterior."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def posterior_variance(self) -> float:
        """Variance of Beta posterior (measures uncertainty in estimate)."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    @property
    def posterior_entropy(self) -> float:
        """Entropy of posterior estimate (uncertainty about hit rate)."""
        return _entropy_bernoulli(self.posterior_mean)
    
    @property
    def avg_relevance(self) -> float:
        """Average relevance score of observed sources."""
        if self.total_sources == 0:
            return 0.5
        return self.sum_relevance / self.total_sources
    
    @property
    def evidence_sufficiency(self) -> float:
        """
        Score [0,1] indicating how sufficient current evidence is.
        
        Based on log-odds aggregation of quality signals:
            sufficiency = sigmoid(l_quotes + l_relevance + l_diversity)
        """
        # Quote coverage factor (log-odds)
        quote_ratio = self.quotes_found / max(1, self.extracts_used)
        l_quotes = _logit(min(0.99, max(0.01, quote_ratio)))
        
        # Relevance factor
        l_relevance = _logit(min(0.99, max(0.01, self.avg_relevance)))
        
        # Source diversity factor (diminishing returns on same domain)
        diversity_ratio = self.relevant_sources / max(1, self.total_sources)
        l_diversity = _logit(min(0.99, max(0.01, diversity_ratio)))
        
        # Combine with equal weights (can be calibrated)
        l_sufficiency = 0.4 * l_quotes + 0.4 * l_relevance + 0.2 * l_diversity
        
        return _sigmoid(l_sufficiency)
    
    def update_from_source(
        self,
        relevance_score: float,
        has_quote: bool,
        is_authoritative: bool = False,
    ) -> None:
        """
        Bayesian update after observing a source.
        
        Updates Beta posterior based on whether source was "useful":
        - Success (α += 1): relevance >= 0.6 AND (has_quote OR is_authoritative)
        - Failure (β += 1): otherwise
        """
        self.total_sources += 1
        self.sum_relevance += relevance_score
        self.max_relevance = max(self.max_relevance, relevance_score)
        
        if relevance_score >= 0.6:
            self.relevant_sources += 1
        
        if has_quote:
            self.quotes_found += 1
        
        if is_authoritative:
            self.authoritative_count += 1
        
        # Determine success for Beta update
        # Using weighted combination (same pattern as evidence_weight in claim_posterior)
        is_useful = (
            relevance_score >= 0.6 and (has_quote or is_authoritative)
        )
        
        if is_useful:
            self.alpha += 1.0
        else:
            self.beta += 0.5  # Partial failure (less weight than success)
    
    def compute_evoi_next_extract(self) -> float:
        """
        Compute Expected Value of Information for next extract.
        
        EVOI = value_uncertainty × posterior_entropy × quality_factor × decay
        
        This is the same formula structure as _expected_value_of_information()
        in target_selection.py.
        """
        params = self.params
        
        # Base value from uncertainty
        value = params.value_uncertainty
        
        # Entropy term: max value when posterior_mean ≈ 0.5 (uncertain)
        # Low value when posterior_mean is extreme (already know hit rate)
        entropy = self.posterior_entropy
        
        # Quality factor based on posterior mean (expected hit rate)
        quality = self.posterior_mean
        
        # Diminishing returns for later extracts
        decay = params.diminishing_returns_decay ** self.extracts_used
        
        # Evidence sufficiency reduces EVOI (already have enough)
        sufficiency_penalty = 1.0 - 0.5 * self.evidence_sufficiency
        
        evoi = value * entropy * quality * decay * sufficiency_penalty
        
        return max(0.0, evoi)
    
    def compute_extract_limit(self) -> int:
        """
        Compute optimal number of extracts using EVOI model.
        
        Returns the k where:
            sum_{i=0}^{k-1} EVOI(i) - cost × k is maximized
            
        """
        params = self.params
        
        # Simulate forward to find optimal k
        optimal_k = self.extracts_used
        cumulative_value = 0.0
        
        for i in range(self.extracts_used, params.max_extracts):
            # Compute marginal EVOI for extract i
            value = params.value_uncertainty
            
            # Entropy decreases as we gather more data
            projected_alpha = self.alpha + (i - self.extracts_used) * self.posterior_mean
            projected_beta = self.beta + (i - self.extracts_used) * (1 - self.posterior_mean)
            projected_mean = projected_alpha / (projected_alpha + projected_beta)
            entropy = _entropy_bernoulli(projected_mean)
            
            # Diminishing returns
            decay = params.diminishing_returns_decay ** i
            
            # EVOI for this extract
            marginal_evoi = value * entropy * projected_mean * decay
            marginal_cost = params.marginal_cost_per_extract
            
            # Stop if marginal value below threshold or cost
            if marginal_evoi < params.min_evoi_threshold:
                break
            if marginal_evoi < marginal_cost:
                break
            
            cumulative_value += marginal_evoi - marginal_cost
            optimal_k = i + 1
        
        # Apply bounds
        result = max(params.min_extracts, min(params.max_extracts, optimal_k))
        
        Trace.event("budget.extract_limit_computed", {
            "extracts_used": self.extracts_used,
            "posterior_mean": round(self.posterior_mean, 4),
            "posterior_entropy": round(self.posterior_entropy, 4),
            "evidence_sufficiency": round(self.evidence_sufficiency, 4),
            "computed_limit": result,
            "cumulative_value": round(cumulative_value, 4),
        })
        
        return result
    
    def should_continue_extracting(self) -> tuple[bool, str]:
        """
        Decide whether to continue extracting based on EVOI analysis.
        
        Returns: (should_continue, reason)
        
        """
        params = self.params
        
        # Hard stop at max
        if self.extracts_used >= params.max_extracts:
            return False, "max_extracts_reached"
        
        # Always do minimum
        if self.extracts_used < params.min_extracts:
            return True, "below_minimum"
        
        # Compute EVOI for next extract
        evoi = self.compute_evoi_next_extract()
        
        # Stop if EVOI below threshold
        if evoi < params.min_evoi_threshold:
            return False, f"evoi_below_threshold:{evoi:.3f}"
        
        # Stop if EVOI below marginal cost
        if evoi < params.marginal_cost_per_extract:
            return False, f"evoi_below_cost:{evoi:.3f}<{params.marginal_cost_per_extract}"
        
        # Stop if evidence is highly sufficient
        if self.evidence_sufficiency > 0.85:
            return False, f"evidence_sufficient:{self.evidence_sufficiency:.3f}"
        
        return True, "continue_evoi_positive"


# -----------------------------------------------------------------------------
# Global Budget Tracker
# -----------------------------------------------------------------------------

@dataclass
class GlobalBudgetTracker:
    """
    Tracks budget state across all claims in a verification run.
    """
    state: BudgetState = field(default_factory=BudgetState)
    claims_processed: int = 0
    total_cost_estimate: float = 0.0
    
    def record_extract(
        self,
        relevance_score: float,
        has_quote: bool,
        is_authoritative: bool = False,
    ) -> None:
        """Record an extract and update Bayesian state."""
        self.state.extracts_used += 1
        self.state.update_from_source(
            relevance_score=relevance_score,
            has_quote=has_quote,
            is_authoritative=is_authoritative,
        )
        self.total_cost_estimate += self.state.params.marginal_cost_per_extract
    
    def get_remaining_budget(self) -> int:
        """Get how many more extracts are worth doing."""
        limit = self.state.compute_extract_limit()
        remaining = limit - self.state.extracts_used
        return max(0, remaining)
    
    def should_extract(self) -> tuple[bool, str]:
        """Check if we should do another extract."""
        return self.state.should_continue_extracting()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize state for logging."""
        return {
            "extracts_used": self.state.extracts_used,
            "quotes_found": self.state.quotes_found,
            "relevant_sources": self.state.relevant_sources,
            "total_sources": self.state.total_sources,
            "posterior_mean": round(self.state.posterior_mean, 4),
            "posterior_entropy": round(self.state.posterior_entropy, 4),
            "evidence_sufficiency": round(self.state.evidence_sufficiency, 4),
            "evoi_next": round(self.state.compute_evoi_next_extract(), 4),
            "computed_limit": self.state.compute_extract_limit(),
            "total_cost_estimate": round(self.total_cost_estimate, 2),
        }


# -----------------------------------------------------------------------------
# Query Budget Functions
# -----------------------------------------------------------------------------

def estimate_claim_complexity(claim: dict) -> float:
    """
    Estimate complexity of a claim for query budget allocation.
    
    Uses same pattern as check_worthiness computation:
    - Text length
    - Verification target type
    - Claim role
    """
    text = claim.get("text", "") or claim.get("normalized_text", "")
    
    # Length factor (log scale, saturates at ~300 chars)
    length_score = min(1.0, math.log1p(len(text)) / math.log1p(300))
    
    # Verification target factor
    target = claim.get("verification_target", "reality")
    target_scores = {
        "attribution": 0.7,  # Need to find who said what
        "reality": 0.5,      # Standard fact check
        "existence": 0.3,    # Just verify something exists
        "none": 0.1,         # No verification needed
    }
    target_score = target_scores.get(target, 0.5)
    
    # Claim role factor
    role = claim.get("claim_role", "support")
    role_scores = {
        "thesis": 0.8,       # Main claim, needs thorough check
        "support": 0.5,      # Supporting evidence
        "background": 0.3,   # Context only
        "hedge": 0.4,        # Qualified statement
        "counterclaim": 0.7, # Opposing view
    }
    role_score = role_scores.get(role, 0.5)
    
    # Combine factors (weighted sum, can be calibrated)
    complexity = 0.3 * length_score + 0.4 * target_score + 0.3 * role_score
    
    return complexity


def has_named_entities(claim: dict) -> bool:
    """
    Check if claim has specific named entities for targeted search.
    
    Heuristic: capitalized words (not at sentence start) or digits.
    """
    text = claim.get("normalized_text", "") or claim.get("text", "")
    
    words = text.split()
    if len(words) < 2:
        return False
    
    # Check for capitalized words (potential entities)
    entity_count = sum(
        1 for i, w in enumerate(words[1:], 1)
        if w and w[0].isupper() and len(w) > 2
    )
    
    # Check for numbers/dates
    has_numbers = any(c.isdigit() for c in text)
    
    return entity_count >= 1 or has_numbers


def compute_query_count(
    claim: dict,
    evidence_sufficiency: float = 0.0,
    params: QueryBudgetParams | None = None,
) -> int:
    """
    Compute optimal number of queries for a claim.
    
    Uses EVOI-like reasoning:
    - Complex claims need more queries (higher potential value)
    - Named entities allow more targeted queries
    - High evidence sufficiency reduces need
    """
    params = params or QueryBudgetParams()
    
    complexity = estimate_claim_complexity(claim)
    has_entities = has_named_entities(claim)
    
    base = params.base_queries
    
    # Complexity adjustment
    if complexity > params.high_complexity_threshold:
        complexity_adj = 1
    elif complexity < params.low_complexity_threshold:
        complexity_adj = -1
    else:
        complexity_adj = 0
    
    # Entity bonus (enables targeted queries)
    entity_adj = 1 if has_entities else 0
    
    # Sufficiency penalty (already have evidence)
    sufficiency_adj = -1 if evidence_sufficiency > 0.7 else 0
    
    result = base + complexity_adj + entity_adj + sufficiency_adj
    result = max(params.min_queries, min(params.max_queries, result))
    
    Trace.event("budget.query_count_computed", {
        "claim_id": claim.get("id"),
        "complexity": round(complexity, 3),
        "has_entities": has_entities,
        "evidence_sufficiency": round(evidence_sufficiency, 3),
        "computed_count": result,
    })
    
    return result

