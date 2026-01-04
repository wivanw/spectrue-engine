# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for evidence_scoring.py module.

Tests the evidence scoring utilities extracted in M119.
"""


from spectrue_core.verification.evidence.evidence_scoring import (
    norm_id,
    is_prob,
    logit,
    sigmoid,
    claim_text,
    explainability_factor_for_tier,
    tier_rank,
    compute_article_g_from_anchor,
    select_anchor_for_article_g,
    TIER_A_PRIOR_MEAN,
    TIER_A_BASELINE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Math Helpers Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNormId:
    """Tests for norm_id function."""

    def test_normalizes_string(self):
        """Should lowercase and strip whitespace."""
        assert norm_id("  C1  ") == "c1"
        assert norm_id("CLAIM_123") == "claim_123"

    def test_handles_none(self):
        """Should handle None input."""
        assert norm_id(None) == ""

    def test_handles_numbers(self):
        """Should convert numbers to strings."""
        assert norm_id(123) == "123"

    def test_handles_empty(self):
        """Should handle empty string."""
        assert norm_id("") == ""


class TestIsProb:
    """Tests for is_prob function."""

    def test_valid_probabilities(self):
        """Should accept valid probabilities [0, 1]."""
        assert is_prob(0.0) is True
        assert is_prob(0.5) is True
        assert is_prob(1.0) is True
        assert is_prob(0.999) is True

    def test_invalid_probabilities(self):
        """Should reject values outside [0, 1]."""
        assert is_prob(-0.1) is False
        assert is_prob(1.1) is False
        assert is_prob(100) is False

    def test_non_numeric(self):
        """Should reject non-numeric values."""
        assert is_prob(None) is False
        assert is_prob("0.5") is False
        assert is_prob([0.5]) is False

    def test_special_floats(self):
        """Should reject NaN and infinity."""
        assert is_prob(float("nan")) is False
        assert is_prob(float("inf")) is False
        assert is_prob(float("-inf")) is False


class TestLogit:
    """Tests for logit function."""

    def test_logit_half(self):
        """logit(0.5) should be 0."""
        assert logit(0.5) == 0.0

    def test_logit_high(self):
        """logit of high probability should be positive."""
        assert logit(0.9) > 0
        assert logit(0.99) > logit(0.9)

    def test_logit_low(self):
        """logit of low probability should be negative."""
        assert logit(0.1) < 0
        assert logit(0.01) < logit(0.1)

    def test_logit_sigmoid_inverse(self):
        """sigmoid(logit(p)) should equal p."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(sigmoid(logit(p)) - p) < 1e-10


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should be 0.5."""
        assert sigmoid(0) == 0.5

    def test_sigmoid_positive(self):
        """sigmoid of positive values should be > 0.5."""
        assert sigmoid(1) > 0.5
        assert sigmoid(10) > 0.99

    def test_sigmoid_negative(self):
        """sigmoid of negative values should be < 0.5."""
        assert sigmoid(-1) < 0.5
        assert sigmoid(-10) < 0.01

    def test_sigmoid_bounds(self):
        """sigmoid should always be in [0, 1]."""
        assert 0 < sigmoid(-10) < 1
        assert 0 < sigmoid(10) < 1
        # Extreme values may round to exactly 0 or 1 due to float precision
        assert 0 <= sigmoid(-100) <= 1
        assert 0 <= sigmoid(100) <= 1


class TestClaimText:
    """Tests for claim_text function."""

    def test_extracts_claim_text_key(self):
        """Should extract from 'claim_text' key."""
        assert claim_text({"claim_text": "Test claim"}) == "Test claim"

    def test_extracts_claim_key(self):
        """Should fallback to 'claim' key."""
        assert claim_text({"claim": "Test claim"}) == "Test claim"

    def test_extracts_text_key(self):
        """Should fallback to 'text' key."""
        assert claim_text({"text": "Test claim"}) == "Test claim"

    def test_priority_order(self):
        """Should prefer claim_text > claim > text."""
        cv = {"claim_text": "A", "claim": "B", "text": "C"}
        assert claim_text(cv) == "A"

    def test_strips_whitespace(self):
        """Should strip whitespace."""
        assert claim_text({"claim_text": "  Test  "}) == "Test"

    def test_handles_empty(self):
        """Should return empty string for missing keys."""
        assert claim_text({}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier-Based Explainability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainabilityFactorForTier:
    """Tests for explainability_factor_for_tier function."""

    def test_tier_a_highest(self):
        """Tier A should have highest factor."""
        factor_a, _, _ = explainability_factor_for_tier("A")
        factor_b, _, _ = explainability_factor_for_tier("B")
        factor_c, _, _ = explainability_factor_for_tier("C")
        assert factor_a > factor_b > factor_c

    def test_tier_a_prime(self):
        """Tier A' should be between A and B."""
        factor_a, _, _ = explainability_factor_for_tier("A")
        factor_a_prime, _, _ = explainability_factor_for_tier("A'")
        factor_b, _, _ = explainability_factor_for_tier("B")
        assert factor_a > factor_a_prime > factor_b

    def test_unknown_tier(self):
        """Unknown tier should return default factor."""
        factor, source, _ = explainability_factor_for_tier("UNKNOWN")
        assert source == "best_tier"
        assert factor < 1.0  # Unknown is below baseline

    def test_none_tier(self):
        """None tier should return unknown_default."""
        factor, source, _ = explainability_factor_for_tier(None)
        assert source == "unknown_default"

    def test_case_insensitive(self):
        """Tier lookup should be case-insensitive."""
        f1, _, _ = explainability_factor_for_tier("A")
        f2, _, _ = explainability_factor_for_tier("a")
        assert f1 == f2


class TestTierRank:
    """Tests for tier_rank function."""

    def test_tier_ordering(self):
        """Tiers should have correct ordering: A > B > C > D."""
        assert tier_rank("A") > tier_rank("B") > tier_rank("C") > tier_rank("D")

    def test_a_and_a_prime_equal(self):
        """A and A' should have equal rank."""
        assert tier_rank("A'") == tier_rank("B")  # Both are 3

    def test_unknown_tier(self):
        """Unknown tier should have rank 0."""
        assert tier_rank("X") == 0
        assert tier_rank(None) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Article G Computation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeArticleGFromAnchor:
    """Tests for compute_article_g_from_anchor function."""

    def test_returns_prior_when_no_anchor(self):
        """Should return prior when no anchor specified."""
        g, debug = compute_article_g_from_anchor(
            anchor_claim_id=None,
            claim_verdicts=[],
            prior_p=0.5,
        )
        assert g == 0.5
        assert debug["used_anchor"] is False

    def test_returns_prior_when_anchor_not_found(self):
        """Should return prior when anchor not in verdicts."""
        g, debug = compute_article_g_from_anchor(
            anchor_claim_id="c99",
            claim_verdicts=[{"claim_id": "c1", "verdict_score": 0.8}],
            prior_p=0.5,
        )
        assert g == 0.5
        assert debug["used_anchor"] is False

    def test_uses_anchor_verdict_score(self):
        """Should compute G using anchor's verdict_score."""
        g, debug = compute_article_g_from_anchor(
            anchor_claim_id="c1",
            claim_verdicts=[{"claim_id": "c1", "verdict_score": 0.8}],
            prior_p=0.5,
        )
        assert debug["used_anchor"] is True
        assert debug["p_anchor"] == 0.8
        assert g > 0.5  # High verdict should push G up

    def test_insufficient_evidence_ignores_anchor(self):
        """Should use k=0 when verdict_state is insufficient_evidence."""
        g, debug = compute_article_g_from_anchor(
            anchor_claim_id="c1",
            claim_verdicts=[{
                "claim_id": "c1",
                "verdict_score": 0.9,
                "verdict_state": "insufficient_evidence",
            }],
            prior_p=0.5,
        )
        assert debug["k"] == 0.0
        assert g == 0.5  # k=0 means G stays at prior

    def test_handles_invalid_verdict_score(self):
        """Should return prior for invalid verdict_score."""
        g, _ = compute_article_g_from_anchor(
            anchor_claim_id="c1",
            claim_verdicts=[{"claim_id": "c1", "verdict_score": "invalid"}],
            prior_p=0.5,
        )
        assert g == 0.5

    def test_handles_boundary_verdict_scores(self):
        """Should return prior for verdict_score at 0 or 1."""
        g0, _ = compute_article_g_from_anchor(
            anchor_claim_id="c1",
            claim_verdicts=[{"claim_id": "c1", "verdict_score": 0.0}],
            prior_p=0.5,
        )
        g1, _ = compute_article_g_from_anchor(
            anchor_claim_id="c1",
            claim_verdicts=[{"claim_id": "c1", "verdict_score": 1.0}],
            prior_p=0.5,
        )
        # Boundary values should return prior (formula invalid)
        assert g0 == 0.5
        assert g1 == 0.5


class TestSelectAnchorForArticleG:
    """Tests for select_anchor_for_article_g function."""

    def test_returns_original_when_no_candidates(self):
        """Should return original anchor when no candidates."""
        anchor, debug = select_anchor_for_article_g(
            anchor_claim_id="c1",
            claim_verdicts=[],
            veracity_debug=[],
        )
        assert anchor == "c1"
        assert debug["reason"] == "no_candidates"

    def test_selects_best_evidence_claim(self):
        """Should select claim with highest evidence-weighted distance."""
        claim_verdicts = [
            {"claim_id": "c1", "verdict_score": 0.9},
            {"claim_id": "c2", "verdict_score": 0.6},
        ]
        veracity_debug = [
            {"claim_id": "c1", "has_direct_evidence": True, "best_tier": "A"},
            {"claim_id": "c2", "has_direct_evidence": True, "best_tier": "B"},
        ]
        anchor, debug = select_anchor_for_article_g(
            anchor_claim_id=None,
            claim_verdicts=claim_verdicts,
            veracity_debug=veracity_debug,
        )
        # c1 has higher score (0.9 vs 0.6) and higher tier (A vs B)
        assert anchor == "c1"
        assert debug["override_used"] is True

    def test_ignores_claims_without_direct_evidence(self):
        """Should skip claims without direct evidence."""
        claim_verdicts = [
            {"claim_id": "c1", "verdict_score": 0.9},
            {"claim_id": "c2", "verdict_score": 0.8},
        ]
        veracity_debug = [
            {"claim_id": "c1", "has_direct_evidence": False, "best_tier": "A"},
            {"claim_id": "c2", "has_direct_evidence": True, "best_tier": "C"},
        ]
        anchor, _ = select_anchor_for_article_g(
            anchor_claim_id=None,
            claim_verdicts=claim_verdicts,
            veracity_debug=veracity_debug,
        )
        # c1 ignored (no direct evidence), c2 selected
        assert anchor == "c2"


# ─────────────────────────────────────────────────────────────────────────────
# Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for module constants."""

    def test_tier_prior_ordering(self):
        """TIER_A_PRIOR_MEAN should have correct ordering."""
        assert TIER_A_PRIOR_MEAN["A"] > TIER_A_PRIOR_MEAN["B"]
        assert TIER_A_PRIOR_MEAN["B"] > TIER_A_PRIOR_MEAN["C"]
        assert TIER_A_PRIOR_MEAN["C"] > TIER_A_PRIOR_MEAN["D"]
        assert TIER_A_PRIOR_MEAN["D"] > TIER_A_PRIOR_MEAN["UNKNOWN"]

    def test_baseline_is_tier_b(self):
        """TIER_A_BASELINE should equal Tier B prior."""
        assert TIER_A_BASELINE == TIER_A_PRIOR_MEAN["B"]

    def test_all_priors_valid_probability(self):
        """All tier priors should be valid probabilities."""
        for tier, prior in TIER_A_PRIOR_MEAN.items():
            assert 0.0 <= prior <= 1.0, f"Invalid prior for tier {tier}"

