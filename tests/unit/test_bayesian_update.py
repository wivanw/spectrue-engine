# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for bayesian_update.py module.

Tests the Bayesian belief update logic extracted in M119.
"""

import pytest
from unittest.mock import patch, MagicMock

from spectrue_core.schema.scoring import BeliefState
from spectrue_core.verification.evidence.bayesian_update import (
    MockEvidence,
    apply_bayesian_update,
)


# ─────────────────────────────────────────────────────────────────────────────
# MockEvidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMockEvidence:
    """Tests for MockEvidence wrapper class."""

    def test_creates_from_dict_with_domain(self):
        """MockEvidence should extract domain from dict."""
        d = {"domain": "example.com", "stance": "support"}
        ev = MockEvidence(d)
        assert ev.domain == "example.com"
        assert ev.stance == "support"

    def test_creates_from_empty_dict(self):
        """MockEvidence should handle empty dict."""
        ev = MockEvidence({})
        assert ev.domain is None
        assert ev.stance is None

    def test_creates_from_partial_dict(self):
        """MockEvidence should handle dict with missing keys."""
        ev = MockEvidence({"domain": "test.org"})
        assert ev.domain == "test.org"
        assert ev.stance is None


# ─────────────────────────────────────────────────────────────────────────────
# apply_bayesian_update Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyBayesianUpdate:
    """Tests for apply_bayesian_update function."""

    @pytest.fixture
    def mock_trace(self):
        """Mock Trace to avoid side effects."""
        with patch("spectrue_core.verification.evidence.bayesian_update.Trace") as mock:
            mock.event = MagicMock()
            mock.event_full = MagicMock()
            yield mock

    @pytest.fixture
    def prior_belief(self):
        """Create a neutral prior belief state."""
        return BeliefState(log_odds=0.0)  # 50% probability

    @pytest.fixture
    def basic_result(self):
        """Create a basic result dict to mutate."""
        return {"verified_score": 0.5}

    def test_returns_trace_dict(self, mock_trace, prior_belief, basic_result):
        """apply_bayesian_update should return a trace dict."""
        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[],
            anchor_claim_id=None,
            importance_by_claim={},
            veracity_debug=[],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.5,
            raw_confidence_score=0.5,
            raw_rationale="Test rationale",
        )

        assert isinstance(trace, dict)
        assert "prior_log_odds" in trace
        assert "consensus_score" in trace
        assert "posterior_log_odds" in trace
        assert "final_probability" in trace
        assert "belief_probability" in trace

    def test_prior_preserved_with_no_evidence(self, mock_trace, prior_belief, basic_result):
        """With no evidence, posterior should be close to prior."""
        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[],
            anchor_claim_id=None,
            importance_by_claim={},
            veracity_debug=[],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.5,
            raw_confidence_score=0.5,
            raw_rationale=None,
        )

        assert trace["prior_log_odds"] == 0.0
        # Posterior should remain near prior with no evidence
        assert abs(trace["posterior_log_odds"]) < 2.0

    def test_handles_claim_verdicts(self, mock_trace, prior_belief, basic_result):
        """Should process claim verdicts for belief update."""
        claim_verdicts = [
            {"claim_id": "c1", "verdict": "supported", "verdict_score": 0.8},
            {"claim_id": "c2", "verdict": "refuted", "verdict_score": 0.7},
        ]

        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=claim_verdicts,
            anchor_claim_id="c1",
            importance_by_claim={"c1": 1.0, "c2": 0.5},
            veracity_debug=[
                {"claim_id": "c1", "veracity_label": "supported"},
                {"claim_id": "c2", "veracity_label": "refuted"},
            ],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.8,
            raw_confidence_score=0.7,
            raw_rationale="Claims analyzed",
        )

        # Should have updated belief
        assert "final_probability" in trace
        assert 0.0 <= trace["final_probability"] <= 1.0

    def test_handles_evidence_pack_as_dict(self, mock_trace, prior_belief, basic_result):
        """Should handle evidence pack as dict."""
        pack = {
            "evidence": [
                {"domain": "reuters.com", "stance": "support"},
                {"domain": "bbc.com", "stance": "support"},
            ]
        }

        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[],
            anchor_claim_id=None,
            importance_by_claim={},
            veracity_debug=[],
            pack=pack,
            result=basic_result,
            raw_verified_score=0.6,
            raw_confidence_score=0.6,
            raw_rationale=None,
        )

        assert "consensus_score" in trace

    def test_handles_evidence_pack_as_object(self, mock_trace, prior_belief, basic_result):
        """Should handle evidence pack as object with .evidence attribute."""
        
        class EvidencePack:
            evidence = [
                {"domain": "nytimes.com", "stance": "neutral"},
            ]

        pack = EvidencePack()

        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[],
            anchor_claim_id=None,
            importance_by_claim={},
            veracity_debug=[],
            pack=pack,
            result=basic_result,
            raw_verified_score=0.5,
            raw_confidence_score=0.5,
            raw_rationale=None,
        )

        assert "consensus_score" in trace

    def test_mutates_result_verified_score(self, mock_trace, prior_belief):
        """Should mutate result['verified_score'] with computed G."""
        result = {"verified_score": 0.99}  # Should be overwritten

        apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[
                {"claim_id": "c1", "verdict": "supported", "verdict_score": 0.7}
            ],
            anchor_claim_id="c1",
            importance_by_claim={"c1": 1.0},
            veracity_debug=[{"claim_id": "c1", "veracity_label": "supported"}],
            pack={"evidence": []},
            result=result,
            raw_verified_score=0.7,
            raw_confidence_score=0.7,
            raw_rationale=None,
        )

        # verified_score should be updated (may or may not equal 0.99)
        assert "verified_score" in result
        assert 0.0 <= result["verified_score"] <= 1.0

    def test_emits_trace_events(self, mock_trace, prior_belief, basic_result):
        """Should emit trace events during update."""
        apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=[
                {"claim_id": "c1", "verdict": "supported", "verdict_score": 0.8}
            ],
            anchor_claim_id="c1",
            importance_by_claim={"c1": 1.0},
            veracity_debug=[{"claim_id": "c1"}],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.8,
            raw_confidence_score=0.8,
            raw_rationale="Test",
        )

        # Should have emitted trace events
        assert mock_trace.event.called or mock_trace.event_full.called

    def test_handles_invalid_verdict_score(self, mock_trace, prior_belief, basic_result):
        """Should handle invalid/missing verdict_score gracefully."""
        claim_verdicts = [
            {"claim_id": "c1", "verdict": "supported", "verdict_score": None},
            {"claim_id": "c2", "verdict": "refuted", "verdict_score": "invalid"},
            {"claim_id": "c3", "verdict": "ambiguous"},  # Missing verdict_score
        ]

        # Should not raise
        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=claim_verdicts,
            anchor_claim_id="c1",
            importance_by_claim={},
            veracity_debug=[],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.5,
            raw_confidence_score=0.5,
            raw_rationale=None,
        )

        assert "final_probability" in trace

    def test_handles_non_list_claim_verdicts(self, mock_trace, prior_belief, basic_result):
        """Should handle non-list claim_verdicts (None, dict, etc.)."""
        # Should not raise with None
        trace = apply_bayesian_update(
            prior_belief=prior_belief,
            context_graph=None,
            claim_verdicts=None,
            anchor_claim_id=None,
            importance_by_claim={},
            veracity_debug=[],
            pack={"evidence": []},
            result=basic_result,
            raw_verified_score=0.5,
            raw_confidence_score=0.5,
            raw_rationale=None,
        )

        assert "final_probability" in trace


# ─────────────────────────────────────────────────────────────────────────────
# BeliefState Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBeliefStateIntegration:
    """Integration tests for BeliefState with Bayesian update."""

    def test_positive_log_odds_yields_high_probability(self):
        """Positive log_odds should yield probability > 0.5."""
        belief = BeliefState(log_odds=2.0)
        # log_odds_to_prob(2.0) ≈ 0.88
        from spectrue_core.scoring.belief import log_odds_to_prob
        prob = log_odds_to_prob(belief.log_odds)
        assert prob > 0.5

    def test_negative_log_odds_yields_low_probability(self):
        """Negative log_odds should yield probability < 0.5."""
        belief = BeliefState(log_odds=-2.0)
        from spectrue_core.scoring.belief import log_odds_to_prob
        prob = log_odds_to_prob(belief.log_odds)
        assert prob < 0.5

    def test_zero_log_odds_yields_half_probability(self):
        """Zero log_odds should yield probability = 0.5."""
        belief = BeliefState(log_odds=0.0)
        from spectrue_core.scoring.belief import log_odds_to_prob
        prob = log_odds_to_prob(belief.log_odds)
        assert abs(prob - 0.5) < 0.001

