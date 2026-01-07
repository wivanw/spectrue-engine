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

import pytest

from spectrue_core.verification.search.search_escalation import (
    QueryVariant,
    EscalationConfig,
    EscalationPass,
    RetrievalOutcome,
    build_query_variants,
    select_topic_from_claim,
    compute_retrieval_outcome,
    should_stop_escalation,
    get_escalation_ladder,
    compute_escalation_reason_codes,
    DEFAULT_ESCALATION_CONFIG,
)


class TestBuildQueryVariants:
    """Tests for build_query_variants()."""

    def test_all_three_variants_with_date(self):
        """Claim with entities, seed terms, explicit date → 3 variants."""
        claim = {
            "subject_entities": ["SpaceX", "Falcon 9"],
            "retrieval_seed_terms": ["launch", "rocket", "orbit", "satellite", "mission"],
            "time_anchor": {
                "type": "explicit_date",
                "value": "2024-01-15",
            },
        }
        variants = build_query_variants(claim)
        
        assert len(variants) == 3
        assert variants[0].query_id == "Q1"
        assert variants[0].strategy == "anchor_tight"
        assert "2024-01-15" in variants[0].text
        
        assert variants[1].query_id == "Q2"
        assert variants[1].strategy == "anchor_medium"
        assert "2024-01-15" not in variants[1].text
        
        assert variants[2].query_id == "Q3"
        assert variants[2].strategy == "broad"

    def test_two_variants_without_date(self):
        """Claim without date anchor → Q1 and Q2 may be the same, only Q2 and Q3."""
        claim = {
            "subject_entities": ["Apple", "iPhone"],
            "retrieval_seed_terms": ["sales", "revenue", "quarter", "market"],
            "time_anchor": {
                "type": "unknown",
            },
        }
        variants = build_query_variants(claim)
        
        # Q1 and Q2 should be identical (no date), so only 2 variants
        assert len(variants) >= 2
        # Check no date in any variant
        for v in variants:
            assert "20" not in v.text  # No year pattern

    def test_max_query_length(self):
        """All variants should be <= 80 chars."""
        claim = {
            "subject_entities": ["VeryLongCompanyNameThatExceedsNormalLength", "AnotherLongEntity"],
            "retrieval_seed_terms": ["term1", "term2", "term3", "term4", "term5", "term6"],
            "time_anchor": {
                "type": "explicit_date",
                "value": "2024-01-15",
            },
        }
        config = EscalationConfig(max_query_length=80)
        variants = build_query_variants(claim, config)
        
        for v in variants:
            assert len(v.text) <= 80, f"Variant {v.query_id} exceeds 80 chars: {len(v.text)}"

    def test_empty_claim(self):
        """Claim with no entities or seed terms → empty list."""
        claim = {}
        variants = build_query_variants(claim)
        assert variants == []

    def test_minimal_claim(self):
        """Claim with minimal fields → at least one variant."""
        claim = {
            "subject_entities": ["Tesla"],
            "retrieval_seed_terms": ["stock", "price"],
        }
        variants = build_query_variants(claim)
        assert len(variants) >= 1
        assert "Tesla" in variants[0].text


class TestSelectTopicFromClaim:
    """Tests for select_topic_from_claim()."""

    def test_reputable_news_falsifiable_by(self):
        """falsifiable_by=reputable_news → topic=news."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "reputable_news",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "falsifiable_by:reputable_news" in reasons

    def test_official_statement_falsifiable_by(self):
        """falsifiable_by=official_statement → topic=news."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "official_statement",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "falsifiable_by:official_statement" in reasons

    def test_dataset_with_explicit_date(self):
        """falsifiable_by=dataset + explicit_date → topic=news."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "dataset",
            },
            "time_anchor": {
                "type": "explicit_date",
                "value": "2024-01-15",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "falsifiable_by:dataset" in reasons
        assert "time_anchor:explicit_date" in reasons

    def test_dataset_without_date(self):
        """falsifiable_by=dataset + unknown time → topic=general."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "dataset",
            },
            "time_anchor": {
                "type": "unknown",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "general"
        assert "falsifiable_by:dataset" in reasons
        assert "time_anchor:unknown" in reasons

    def test_scientific_publication_range(self):
        """falsifiable_by=scientific_publication + range → topic=news."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "scientific_publication",
            },
            "time_anchor": {
                "type": "range",
                "start": "2023",
                "end": "2024",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"

    def test_default_news(self):
        """No falsifiability → default topic=news."""
        claim = {}
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "default_news" in reasons or "no_falsifiability" in reasons


class TestComputeRetrievalOutcome:
    """Tests for compute_retrieval_outcome()."""

    def test_empty_sources(self):
        """Empty sources list → all zeros."""
        outcome = compute_retrieval_outcome([])
        
        assert outcome.sources_count == 0
        assert outcome.best_relevance == 0.0
        assert outcome.usable_snippets_count == 0
        assert outcome.match_accept_count == 0

    def test_sources_with_support_stance(self):
        """Sources with SUPPORT stance → counted in match_accept."""
        sources = [
            {"url": "https://example.com/a", "stance": "SUPPORT", "snippet": "a" * 60, "score": 0.8},
            {"url": "https://example.com/b", "stance": "context", "snippet": "b" * 60, "score": 0.5},
        ]
        outcome = compute_retrieval_outcome(sources)
        
        assert outcome.sources_count == 2
        assert outcome.match_accept_count == 1  # Only SUPPORT
        assert outcome.usable_snippets_count == 2  # Both have 60+ chars
        assert outcome.best_relevance == 0.8

    def test_sources_with_quote_matches(self):
        """Sources with quote_matches=True → counted in match_accept."""
        sources = [
            {"url": "https://example.com/a", "quote_matches": True, "snippet": "a" * 60},
            {"url": "https://example.com/b", "quote_matches": False, "snippet": "b" * 60},
        ]
        outcome = compute_retrieval_outcome(sources)
        
        assert outcome.match_accept_count == 1

    def test_short_snippets_not_counted(self):
        """Snippets < 50 chars not counted as usable."""
        sources = [
            {"url": "https://example.com/a", "snippet": "short"},
            {"url": "https://example.com/b", "snippet": "x" * 100},
        ]
        outcome = compute_retrieval_outcome(sources)
        
        assert outcome.usable_snippets_count == 1


class TestShouldStopEscalation:
    """Tests for should_stop_escalation()."""

    def test_stop_on_match_accept(self):
        """match_accept_count >= 1 → stop."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.5,
            usable_snippets_count=1,
            match_accept_count=1,
        )
        should_stop, reason = should_stop_escalation(outcome)
        
        assert should_stop is True
        assert reason == "match_accept"

    def test_stop_on_snippets_and_relevance(self):
        """usable_snippets >= 2 AND relevance >= 0.2 → stop."""
        outcome = RetrievalOutcome(
            sources_count=5,
            best_relevance=0.3,
            usable_snippets_count=2,
            match_accept_count=0,
        )
        config = EscalationConfig(min_usable_snippets=2, min_relevance_threshold=0.2)
        should_stop, reason = should_stop_escalation(outcome, config)
        
        assert should_stop is True
        assert reason == "snippets_and_relevance"

    def test_continue_low_relevance(self):
        """Snippets ok but relevance low → continue."""
        outcome = RetrievalOutcome(
            sources_count=5,
            best_relevance=0.1,  # Below threshold
            usable_snippets_count=3,
            match_accept_count=0,
        )
        config = EscalationConfig(min_relevance_threshold=0.2)
        should_stop, reason = should_stop_escalation(outcome, config)
        
        assert should_stop is False
        assert reason == "continue"

    def test_continue_insufficient_snippets(self):
        """Relevance ok but snippets low → continue."""
        outcome = RetrievalOutcome(
            sources_count=5,
            best_relevance=0.5,
            usable_snippets_count=1,  # Below threshold
            match_accept_count=0,
        )
        config = EscalationConfig(min_usable_snippets=2)
        should_stop, reason = should_stop_escalation(outcome, config)
        
        assert should_stop is False
        assert reason == "continue"


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
        assert pass_c.max_results == 6

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
        outcome = RetrievalOutcome(
            sources_count=0,
            best_relevance=0.0,
            usable_snippets_count=0,
            match_accept_count=0,
        )
        reasons = compute_escalation_reason_codes(outcome)
        
        assert "no_sources" in reasons

    def test_no_snippets(self):
        """Sources but no usable snippets → 'no_snippets' reason."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.5,
            usable_snippets_count=0,
            match_accept_count=0,
        )
        reasons = compute_escalation_reason_codes(outcome)
        
        assert "no_snippets" in reasons

    def test_low_relevance(self):
        """Relevance below 0.2 → 'low_relevance' reason."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.1,
            usable_snippets_count=2,
            match_accept_count=0,
        )
        reasons = compute_escalation_reason_codes(outcome)
        
        assert "low_relevance" in reasons
