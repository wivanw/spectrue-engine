# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for ClaimPosteriorModel (Phase 2: Bayesian Calibration).

Tests the unified posterior computation that replaces double-counting.
"""

from spectrue_core.scoring.claim_posterior import (
    compute_claim_posterior,
    aggregate_article_posterior,
    EvidenceItem,
    PosteriorParams,
    _logit,
    _sigmoid,
    _stance_direction,
)


class TestLogOddsConversion:
    """Test logit/sigmoid roundtrip."""
    
    def test_logit_sigmoid_roundtrip(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-9
    
    def test_logit_at_half(self):
        assert abs(_logit(0.5)) < 1e-9  # logit(0.5) = 0
    
    def test_sigmoid_at_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9  # sigmoid(0) = 0.5


class TestStanceDirection:
    """Test stance to direction mapping."""
    
    def test_support_positive(self):
        assert _stance_direction("support") == 1.0
        assert _stance_direction("SUPPORT") == 1.0
        assert _stance_direction("sup") == 1.0
    
    def test_refute_negative(self):
        assert _stance_direction("refute") == -1.0
        assert _stance_direction("REFUTE") == -1.0
        assert _stance_direction("contradict") == -1.0
    
    def test_neutral_zero(self):
        assert _stance_direction("neutral") == 0.0
        assert _stance_direction("context") == 0.0
        assert _stance_direction("irrelevant") == 0.0


class TestPosteriorComputation:
    """Test the main posterior computation."""
    
    def test_neutral_prior_neutral_evidence(self):
        """No evidence, neutral LLM -> stays at prior."""
        result = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier=None,
            evidence_items=[],
        )
        # With default params: prior=0.5, llm=0.5, evidence=0
        # l_post = 0 + 1*0 + 1*0 = 0 -> p = 0.5
        assert 0.45 < result.p_posterior < 0.55
    
    def test_strong_support_evidence(self):
        """Support evidence should increase posterior."""
        items = [
            EvidenceItem(stance="support", tier="B", relevance=0.8, quote_present=True),
        ]
        result = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="B",
            evidence_items=items,
        )
        # Should be > 0.5 due to support evidence
        assert result.p_posterior > 0.6
        assert result.n_support == 1
        assert result.n_refute == 0
    
    def test_strong_refute_evidence(self):
        """Refute evidence should decrease posterior."""
        items = [
            EvidenceItem(stance="refute", tier="B", relevance=0.8, quote_present=True),
        ]
        result = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="B",
            evidence_items=items,
        )
        # Should be < 0.5 due to refute evidence
        assert result.p_posterior < 0.4
        assert result.n_support == 0
        assert result.n_refute == 1
    
    def test_conflicting_evidence(self):
        """Support and refute cancel out."""
        items = [
            EvidenceItem(stance="support", tier="B", relevance=0.8, quote_present=True),
            EvidenceItem(stance="refute", tier="B", relevance=0.8, quote_present=True),
        ]
        result = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="B",
            evidence_items=items,
        )
        # Should be close to 0.5 (tier B prior is 0.6, so slightly above)
        assert 0.4 < result.p_posterior < 0.7
        assert result.n_support == 1
        assert result.n_refute == 1
    
    def test_llm_confidence_shifts_posterior(self):
        """High LLM score should shift posterior up."""
        result_low = compute_claim_posterior(
            llm_verdict_score=0.3,
            best_tier="C",
            evidence_items=[],
        )
        result_high = compute_claim_posterior(
            llm_verdict_score=0.7,
            best_tier="C",
            evidence_items=[],
        )
        assert result_high.p_posterior > result_low.p_posterior
    
    def test_tier_affects_prior(self):
        """Higher tier should give higher prior."""
        result_tier_a = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="A",
            evidence_items=[],
        )
        result_tier_d = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="D",
            evidence_items=[],
        )
        # Tier A has higher prior (0.7) than D (0.4)
        assert result_tier_a.p_posterior > result_tier_d.p_posterior


class TestPosteriorParams:
    """Test custom calibration parameters."""
    
    def test_alpha_zero_ignores_llm(self):
        """With alpha=0, LLM score should not affect posterior."""
        params = PosteriorParams(alpha=0.0, beta=1.0)
        result_low = compute_claim_posterior(
            llm_verdict_score=0.1,
            best_tier="C",
            evidence_items=[],
            params=params,
        )
        result_high = compute_claim_posterior(
            llm_verdict_score=0.9,
            best_tier="C",
            evidence_items=[],
            params=params,
        )
        # With alpha=0, both should be equal (just prior)
        assert abs(result_low.p_posterior - result_high.p_posterior) < 0.01
    
    def test_beta_zero_ignores_evidence(self):
        """With beta=0, evidence should not affect posterior."""
        params = PosteriorParams(alpha=1.0, beta=0.0)
        items = [
            EvidenceItem(stance="support", tier="A", relevance=1.0, quote_present=True),
            EvidenceItem(stance="support", tier="A", relevance=1.0, quote_present=True),
        ]
        result_no_ev = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="C",
            evidence_items=[],
            params=params,
        )
        result_with_ev = compute_claim_posterior(
            llm_verdict_score=0.5,
            best_tier="C",
            evidence_items=items,
            params=params,
        )
        # With beta=0, both should be equal
        assert abs(result_no_ev.p_posterior - result_with_ev.p_posterior) < 0.01


class TestArticleAggregation:
    """Test article-level aggregation."""
    
    def test_max_aggregation(self):
        """Max should return most extreme posterior."""
        posteriors = [("c1", 0.5), ("c2", 0.8), ("c3", 0.6)]
        result = aggregate_article_posterior(posteriors, method="max")
        assert result == 0.8  # Furthest from 0.5
    
    def test_mean_aggregation(self):
        """Mean should return average."""
        posteriors = [("c1", 0.4), ("c2", 0.6), ("c3", 0.8)]
        result = aggregate_article_posterior(posteriors, method="mean")
        assert abs(result - 0.6) < 0.01
    
    def test_empty_returns_neutral(self):
        """Empty list should return 0.5."""
        result = aggregate_article_posterior([], method="max")
        assert result == 0.5


class TestInvariant_NoDoubleCount:
    """
    Critical invariant: weak evidence should NOT inflate score to tier-prior level.
    
    This was the bug we're fixing - evidence counted twice led to inflation.
    """
    
    def test_weak_evidence_no_inflation(self):
        """
        With weak/neutral evidence, posterior should not jump to tier prior.
        
        Old bug: LLM=0.5, weak evidence -> aggregate gave 0.85 -> anchor formula
        gave another boost -> final was artificially high.
        
        New: single formula, weak evidence contributes little.
        """
        # Weak neutral evidence
        items = [
            EvidenceItem(stance="context", tier="C", relevance=0.3, quote_present=False),
        ]
        result = compute_claim_posterior(
            llm_verdict_score=0.5,  # Ambiguous LLM
            best_tier="B",  # B tier prior is 0.6
            evidence_items=items,
        )
        # Should NOT be inflated to 0.85+
        # Should be close to blend of prior (0.6) and LLM (0.5)
        assert result.p_posterior < 0.75
        # Evidence log-odds should be ~0 for neutral stance
        assert abs(result.log_odds_evidence) < 0.5

