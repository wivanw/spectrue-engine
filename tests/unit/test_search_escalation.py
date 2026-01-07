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

from spectrue_core.verification.search.search_escalation import (
    EscalationConfig,
    RetrievalOutcome,
    build_query_variants,
    select_topic_from_claim,
    compute_retrieval_outcome,
    should_stop_escalation,
    get_escalation_ladder,
    compute_escalation_reason_codes,
    _deduplicate_tokens,
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
            "subject_entities": ["VeryLongCompanyName", "AnotherLongEntity"],
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

    def test_token_deduplication(self):
        """Tokens should be deduplicated in queries."""
        claim = {
            "subject_entities": ["SpaceX", "Rocket"],
            "retrieval_seed_terms": ["spacex", "rocket", "launch"],  # duplicates with different case
        }
        variants = build_query_variants(claim)
        
        # Should deduplicate (spacex appears once, rocket appears once)
        for v in variants:
            tokens = v.text.lower().split()
            # No exact duplicates (normalized)
            assert len(tokens) == len(set(tokens)), f"Duplicates in {v.text}"

    def test_long_phrase_filtered(self):
        """Multi-word phrases (>3 words) should be filtered from seed terms."""
        claim = {
            "subject_entities": ["Tesla"],
            "retrieval_seed_terms": [
                "stock",
                "this is a very long phrase that should be filtered",  # >3 words
                "price",
            ],
        }
        variants = build_query_variants(claim)
        
        for v in variants:
            assert "very long phrase" not in v.text

    def test_no_entities_uses_seed_terms_for_q3(self):
        """If entities empty, Q3 should use seed terms only."""
        claim = {
            "subject_entities": [],
            "retrieval_seed_terms": ["launch", "rocket", "orbit", "mission"],
        }
        variants = build_query_variants(claim)
        
        # Should still generate variants from seed terms
        assert len(variants) >= 1
        # Q3 should have some seed terms
        q3 = next((v for v in variants if v.query_id == "Q3"), None)
        if q3:
            assert "launch" in q3.text.lower() or "rocket" in q3.text.lower()


class TestDeduplicateTokens:
    """Tests for _deduplicate_tokens()."""

    def test_removes_duplicates(self):
        """Should remove duplicate tokens preserving order."""
        tokens = ["Apple", "apple", "iPhone", "APPLE"]
        result = _deduplicate_tokens(tokens)
        assert len(result) == 2  # Apple and iPhone
        assert result[0] == "Apple"  # Original case preserved
        assert result[1] == "iPhone"

    def test_preserves_order(self):
        """Should preserve original order."""
        tokens = ["first", "second", "third"]
        result = _deduplicate_tokens(tokens)
        assert result == ["first", "second", "third"]

    def test_filters_short_tokens(self):
        """Should filter tokens with length < 2."""
        tokens = ["a", "ab", "abc"]
        result = _deduplicate_tokens(tokens)
        assert result == ["ab", "abc"]

    def test_filters_long_tokens(self):
        """Should filter tokens exceeding max_token_length."""
        tokens = ["short", "x" * 50]
        result = _deduplicate_tokens(tokens, max_token_length=30)
        assert result == ["short"]


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

    def test_default_news_no_falsifiability(self):
        """No falsifiability → default topic=news with diagnostic."""
        claim = {}
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "no_falsifiability_field" in reasons
        assert "default_news" in reasons

    def test_default_news_other_falsifiable_by(self):
        """falsifiable_by=other → default news with diagnostic."""
        claim = {
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "other",
            },
        }
        topic, reasons = select_topic_from_claim(claim)
        
        assert topic == "news"
        assert "falsifiable_by:other" in reasons
        assert "unclassified_falsifiable_by" in reasons
        assert "default_news" in reasons


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
        """usable_snippets >= 2 AND relevance >= 0.2 → stop."""
        outcome = RetrievalOutcome(
            sources_count=5,
            best_relevance=0.3,
            usable_snippets_count=2,
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
        )
        reasons = compute_escalation_reason_codes(outcome)
        
        assert "no_sources" in reasons

    def test_no_snippets(self):
        """Sources but no usable snippets → 'no_snippets' reason."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.5,
            usable_snippets_count=0,
        )
        reasons = compute_escalation_reason_codes(outcome)
        
        assert "no_snippets" in reasons

    def test_low_relevance_uses_config_threshold(self):
        """Relevance below config threshold → low_relevance with threshold in reason."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.1,
            usable_snippets_count=2,
        )
        config = EscalationConfig(min_relevance_threshold=0.3)
        reasons = compute_escalation_reason_codes(outcome, config)
        
        # Should include threshold in reason for diagnostics
        assert any("low_relevance" in r for r in reasons)
        assert any("0.3" in r for r in reasons)  # Threshold included

    def test_insufficient_snippets_uses_config_threshold(self):
        """Snippets below config threshold → reason includes threshold."""
        outcome = RetrievalOutcome(
            sources_count=3,
            best_relevance=0.5,
            usable_snippets_count=1,
        )
        config = EscalationConfig(min_usable_snippets=3)
        reasons = compute_escalation_reason_codes(outcome, config)
        
        assert any("insufficient_snippets" in r for r in reasons)
        assert any("1<3" in r for r in reasons)  # Current < threshold
