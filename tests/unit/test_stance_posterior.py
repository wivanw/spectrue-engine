"""
Unit tests for Bayesian Stance Posterior Estimator.
"""

import pytest

from spectrue_core.verification.stance_posterior import (
    StanceFeatures,
    StancePosterior,
    compute_stance_posterior,
    compute_effective_evidence_count,
    source_prior_from_tier,
    STANCE_SUPPORT,
    STANCE_REFUTE,
    STANCE_NEUTRAL,
)


class TestStanceFeatures:
    """Test StanceFeatures dataclass."""

    def test_basic_creation(self):
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=0.8,
            quote_present=True,
            has_evidence_chunk=True,
            source_prior=0.7,
        )
        assert features.llm_stance == "SUPPORT"
        assert features.llm_relevance == 0.8
        assert features.quote_present is True


class TestSourcePriorFromTier:
    """Test tier to prior conversion."""

    def test_tier_a(self):
        assert source_prior_from_tier("A") == 0.9
        assert source_prior_from_tier("a") == 0.9

    def test_tier_b(self):
        assert source_prior_from_tier("B") == 0.7

    def test_tier_c(self):
        assert source_prior_from_tier("C") == 0.5

    def test_tier_d(self):
        assert source_prior_from_tier("D") == 0.3

    def test_unknown(self):
        assert source_prior_from_tier(None) == 0.5
        assert source_prior_from_tier("X") == 0.5


class TestComputeStancePosterior:
    """Test Bayesian posterior calculation."""

    def test_strong_support_signal(self):
        """LLM says SUPPORT + quote present + high relevance => high p_support."""
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=0.9,
            quote_present=True,
            has_evidence_chunk=True,
            source_prior=0.7,
        )
        posterior = compute_stance_posterior(features)
        
        assert posterior.p_support > 0.5, "Strong support signals should yield p_support > 0.5"
        assert posterior.argmax_stance == STANCE_SUPPORT
        assert posterior.p_evidence > 0.5

    def test_weak_support_no_quote(self):
        """LLM says SUPPORT but no quote => lower p_support."""
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=0.3,
            quote_present=False,
            has_evidence_chunk=False,
            source_prior=0.5,
        )
        posterior = compute_stance_posterior(features)
        
        # Should still have some support probability but lower
        assert posterior.p_support < 0.5, "Weak signals should yield lower p_support"

    def test_strong_refute_signal(self):
        """LLM says REFUTE + quote + high relevance => high p_refute."""
        features = StanceFeatures(
            llm_stance="REFUTE",
            llm_relevance=0.85,
            quote_present=True,
            has_evidence_chunk=True,
            source_prior=0.8,
        )
        posterior = compute_stance_posterior(features)
        
        assert posterior.p_refute > posterior.p_support
        assert posterior.argmax_stance == STANCE_REFUTE

    def test_context_default(self):
        """LLM says CONTEXT, no strong signals => argmax = context."""
        features = StanceFeatures(
            llm_stance="CONTEXT",
            llm_relevance=0.4,
            quote_present=False,
            has_evidence_chunk=True,
            source_prior=0.5,
        )
        posterior = compute_stance_posterior(features)
        
        # Without strong stance signals, neutral/context should dominate
        assert posterior.p_evidence < 0.5

    def test_missing_relevance(self):
        """Missing relevance should default to 0.5."""
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=None,  # Missing
            quote_present=True,
            has_evidence_chunk=True,
            source_prior=0.5,
        )
        posterior = compute_stance_posterior(features)
        
        # Should still work, using default 0.5 relevance
        assert 0 <= posterior.p_support <= 1
        assert posterior.p_support > 0.3  # Quote present gives boost

    def test_probabilities_sum_to_one(self):
        """Posterior probabilities should sum to 1."""
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=0.6,
            quote_present=True,
            has_evidence_chunk=True,
            source_prior=0.7,
        )
        posterior = compute_stance_posterior(features)
        
        total = (
            posterior.p_support +
            posterior.p_refute +
            posterior.p_neutral +
            posterior.p_context +
            posterior.p_irrelevant
        )
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected 1.0"

    def test_entropy_range(self):
        """Entropy should be non-negative."""
        features = StanceFeatures(
            llm_stance="SUPPORT",
            llm_relevance=0.5,
            quote_present=False,
            has_evidence_chunk=True,
            source_prior=0.5,
        )
        posterior = compute_stance_posterior(features)
        
        assert posterior.entropy >= 0


class TestComputeEffectiveEvidenceCount:
    """Test expected evidence count calculation."""

    def test_single_strong_support(self):
        """Single source with high p_support."""
        posteriors = [
            StancePosterior(
                p_support=0.8,
                p_refute=0.05,
                p_neutral=0.1,
                p_context=0.03,
                p_irrelevant=0.02,
                p_evidence=0.85,
                argmax_stance="support",
                entropy=0.5,
            )
        ]
        counts = compute_effective_evidence_count(posteriors)
        
        assert counts["effective_support"] == 0.8
        assert counts["effective_evidence"] == 0.85
        assert counts["total_sources"] == 1

    def test_multiple_sources(self):
        """Multiple sources with varying posteriors."""
        posteriors = [
            StancePosterior(
                p_support=0.6, p_refute=0.1, p_neutral=0.2,
                p_context=0.05, p_irrelevant=0.05,
                p_evidence=0.7, argmax_stance="support", entropy=0.8,
            ),
            StancePosterior(
                p_support=0.3, p_refute=0.5, p_neutral=0.1,
                p_context=0.05, p_irrelevant=0.05,
                p_evidence=0.8, argmax_stance="refute", entropy=0.9,
            ),
            StancePosterior(
                p_support=0.1, p_refute=0.1, p_neutral=0.5,
                p_context=0.2, p_irrelevant=0.1,
                p_evidence=0.2, argmax_stance="neutral", entropy=1.2,
            ),
        ]
        counts = compute_effective_evidence_count(posteriors)
        
        assert counts["effective_support"] == 1.0  # 0.6 + 0.3 + 0.1
        assert counts["effective_refute"] == 0.7   # 0.1 + 0.5 + 0.1
        assert counts["total_sources"] == 3

    def test_empty_list(self):
        """Empty posteriors list."""
        counts = compute_effective_evidence_count([])
        
        assert counts["effective_support"] == 0.0
        assert counts["total_sources"] == 0


class TestPosteriorToDict:
    """Test serialization."""

    def test_to_dict(self):
        posterior = StancePosterior(
            p_support=0.6,
            p_refute=0.1,
            p_neutral=0.2,
            p_context=0.05,
            p_irrelevant=0.05,
            p_evidence=0.7,
            argmax_stance="support",
            entropy=0.8,
        )
        d = posterior.to_dict()
        
        assert d["p_support"] == 0.6
        assert d["argmax_stance"] == "support"
        assert "p_evidence" in d
