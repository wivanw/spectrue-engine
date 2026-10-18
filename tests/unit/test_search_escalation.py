# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""
Unit tests for M126 search escalation module.
"""

from spectrue_core.use_cases.verification.search.search_escalation import (
    EscalationConfig,
    compute_retrieval_outcome,
    should_stop_escalation,
    get_escalation_ladder,
    compute_escalation_reason_codes,
)


class TestComputeRetrievalOutcome:
    """Tests for compute_retrieval_outcome()."""

    def test_empty_sources(self):
        """Empty sources list → all zeros."""
        outcome = compute_retrieval_outcome([])
        
        assert outcome.sources_count == 0
        assert outcome.best_relevance == 0.0
        assert outcome.usable_snippets_count == 0

    def test_sources_with_relevance(self):
        """Sources with score → best_relevance computed."""
        sources = [
            {"url": "https://example.com/a", "snippet": "a" * 60, "score": 0.8},
            {"url": "https://example.com/b", "snippet": "b" * 60, "score": 0.5},
        ]
        outcome = compute_retrieval_outcome(sources)
        
        assert outcome.sources_count == 2
        assert outcome.usable_snippets_count == 2  # Both have 60+ chars
        assert outcome.best_relevance == 0.8

    def test_short_snippets_not_counted(self):
        """Snippets < min_snippet_chars not counted as usable."""
        sources = [
            {"url": "https://example.com/a", "snippet": "short"},
            {"url": "https://example.com/b", "snippet": "x" * 100},
        ]
        outcome = compute_retrieval_outcome(sources)
        
        assert outcome.usable_snippets_count == 1

    def test_configurable_snippet_threshold(self):
        """min_snippet_chars from config is respected."""
        sources = [
            {"url": "https://example.com/a", "snippet": "x" * 30},
            {"url": "https://example.com/b", "snippet": "x" * 100},
        ]
        # Default threshold (50)
        outcome_default = compute_retrieval_outcome(sources)
        assert outcome_default.usable_snippets_count == 1
        
        # Lower threshold (20)
        config = EscalationConfig(min_snippet_chars=20)
        outcome_low = compute_retrieval_outcome(sources, config)
        assert outcome_low.usable_snippets_count == 2

    def test_no_match_accept_count(self):
        """RetrievalOutcome should NOT have match_accept_count (removed)."""
        outcome = compute_retrieval_outcome([])
        assert not hasattr(outcome, "match_accept_count")


class TestShouldStopEscalation:
    """Tests for should_stop_escalation()."""

    def test_stop_on_snippets_and_relevance(self):
        """usable_snippets >= 2 AND relevance >= 0.2 → stop (technical fallback)."""
        claim = {"id": "c1", "text": "Test"}
        # 2 sources with good relevance and snippets
        sources = [
            {"url": "https://a.com", "score": 0.3, "snippet": "x" * 60},
            {"url": "https://b.com", "score": 0.3, "snippet": "y" * 60},
        ]
        config = EscalationConfig(min_usable_snippets=2, min_relevance_threshold=0.2)
        should_stop, reason = should_stop_escalation(claim, sources, config)
        
        assert should_stop is True
        assert reason == "snippets_and_relevance"

    def test_continue_low_relevance(self):
        """Snippets ok but relevance low → continue."""
        claim = {"id": "c1", "text": "Test"}
        sources = [
            {"url": "https://a.com", "score": 0.1, "snippet": "x" * 60},
            {"url": "https://b.com", "score": 0.1, "snippet": "y" * 60},
        ]
        config = EscalationConfig(min_relevance_threshold=0.2)
        should_stop, reason = should_stop_escalation(claim, sources, config)
        
        assert should_stop is False
        assert reason == "insufficient"

    def test_continue_insufficient_snippets(self):
        """Relevance ok but snippets low → continue."""
        claim = {"id": "c1", "text": "Test"}
        sources = [
            {"url": "https://a.com", "score": 0.5, "snippet": "short"},
            {"url": "https://b.com", "score": 0.5, "snippet": "short"},
        ]
        config = EscalationConfig(min_usable_snippets=2)
        should_stop, reason = should_stop_escalation(claim, sources, config)
        
        assert should_stop is False
        assert reason == "insufficient"


class TestEscalationLadder:
    """Tests for get_escalation_ladder()."""

    def test_ladder_has_four_passes(self):
        """Ladder should have A, B, C, D passes."""
        ladder = get_escalation_ladder()
        
        assert len(ladder) == 4
        pass_ids = [p.pass_id for p in ladder]
        assert pass_ids == ["A", "B", "C", "D"]

    def test_pass_a_is_cheap(self):
        """Pass A should be cheap: basic depth, 3 results."""
        ladder = get_escalation_ladder()
        pass_a = ladder[0]
        
        assert pass_a.search_depth == "basic"
        assert pass_a.max_results == 3
        assert pass_a.include_domains_relaxed is False

    def test_pass_c_is_advanced(self):
        """Pass C should use advanced depth."""
        ladder = get_escalation_ladder()
        pass_c = ladder[2]
        
        assert pass_c.search_depth == "advanced"
        assert pass_c.max_results == 5

    def test_pass_d_relaxes_domains(self):
        """Pass D should relax domain restrictions."""
        ladder = get_escalation_ladder()
        pass_d = ladder[3]
        
        assert pass_d.include_domains_relaxed is True
        assert pass_d.search_depth == "basic"  # Not expensive depth


class TestComputeEscalationReasonCodes:
    """Tests for compute_escalation_reason_codes()."""

    def test_no_sources(self):
        """No sources → 'no_sources' reason."""
        claim = {"id": "c1", "text": "Test"}
        sources = []
        outcome = compute_retrieval_outcome(sources)
        reasons = compute_escalation_reason_codes(claim, sources, outcome)
        
        assert "no_sources" in reasons

    def test_no_snippets(self):
        """Sources but no usable snippets → 'no_snippets' reason."""
        claim = {"id": "c1", "text": "Test"}
        sources = [{"url": "https://a.com", "score": 0.5, "snippet": ""}]
        outcome = compute_retrieval_outcome(sources)
        reasons = compute_escalation_reason_codes(claim, sources, outcome)
        
        assert "no_snippets" in reasons

    def test_low_relevance_uses_config_threshold(self):
        """Relevance below config threshold → low_relevance with threshold in reason."""
        claim = {"id": "c1", "text": "Test"}
        sources = [{"url": "https://a.com", "score": 0.1, "snippet": "x" * 60}]
        config = EscalationConfig(min_relevance_threshold=0.3)
        outcome = compute_retrieval_outcome(sources, config)
        reasons = compute_escalation_reason_codes(claim, sources, outcome, config)
        
        # Should include threshold in reason for diagnostics
        assert any("low_relevance" in r for r in reasons)
        assert any("0.3" in r for r in reasons)  # Threshold included

    def test_insufficient_snippets_uses_config_threshold(self):
        """Snippets below config threshold → reason includes threshold."""
        claim = {"id": "c1", "text": "Test"}
        sources = [{"url": "https://a.com", "score": 0.5, "snippet": "x" * 60}]
        config = EscalationConfig(min_usable_snippets=3)
        outcome = compute_retrieval_outcome(sources, config)
        reasons = compute_escalation_reason_codes(claim, sources, outcome, config)
        
        assert any("insufficient_snippets" in r for r in reasons)
        assert any("1<3" in r for r in reasons)  # Current < threshold
