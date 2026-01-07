# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
M125: Tests for validate_core_claim() - verifiability contract enforcement.
"""

from spectrue_core.agents.skills.claims import (
    validate_core_claim,
    ExtractionStats,
    TIME_ANCHOR_EXEMPT_PREDICATES,
)


class TestValidateCoreClaimAccept:
    """Test cases where claims should be accepted (valid)."""

    def test_accept_well_formed_event_claim(self):
        """A well-formed event claim with all anchors should be accepted."""
        claim = {
            "claim_text": "Tesla announced record quarterly earnings of $5 billion in Q4 2024.",
            "normalized_text": "Tesla announced record quarterly earnings of $5 billion in Q4 2024.",
            "subject_entities": ["Tesla"],
            "predicate_type": "measurement",
            "time_anchor": {"type": "explicit_date", "value": "Q4 2024"},
            "location_anchor": "unknown",
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "official_statement",
            },
            "retrieval_seed_terms": ["Tesla", "earnings", "Q4", "2024", "billion"],
            "importance": 0.9,
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is True
        assert reason_codes == []

    def test_accept_quote_claim_with_relative_time(self):
        """A quote claim with relative time anchor should be accepted."""
        claim = {
            "claim_text": "Elon Musk said yesterday that SpaceX will launch Starship.",
            "subject_entities": ["Elon Musk", "SpaceX", "Starship"],
            "predicate_type": "quote",
            "time_anchor": {"type": "relative", "value": "yesterday"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "reputable_news",
            },
            "retrieval_seed_terms": ["Elon", "Musk", "SpaceX", "Starship", "launch"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is True
        assert reason_codes == []

    def test_accept_policy_claim_without_time_anchor(self):
        """Policy claims don't require explicit time anchors."""
        claim = {
            "claim_text": "The EU requires all new cars to have automatic emergency braking.",
            "subject_entities": ["European Union"],
            "predicate_type": "policy",
            "time_anchor": {"type": "unknown", "value": "unknown"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "public_records",
            },
            "retrieval_seed_terms": ["EU", "cars", "emergency", "braking", "regulation"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is True, f"Policy claim should be accepted even with unknown time anchor: {reason_codes}"
        assert reason_codes == []

    def test_accept_ranking_claim_without_time_anchor(self):
        """Ranking claims don't require explicit time anchors."""
        claim = {
            "claim_text": "Apple is the world's most valuable company.",
            "subject_entities": ["Apple"],
            "predicate_type": "ranking",
            "time_anchor": {"type": "unknown", "value": "unknown"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "dataset",
            },
            "retrieval_seed_terms": ["Apple", "valuable", "company", "market", "cap"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is True
        assert reason_codes == []


class TestValidateCoreClaimReject:
    """Test cases where claims should be rejected (invalid)."""

    def test_reject_claim_with_is_falsifiable_false(self):
        """Claims with is_falsifiable=false should be rejected."""
        claim = {
            "claim_text": "This is an encouraging sign for the economy.",
            "subject_entities": ["economy"],
            "predicate_type": "other",
            "time_anchor": {"type": "relative", "value": "now"},
            "falsifiability": {
                "is_falsifiable": False,
                "falsifiable_by": "other",
            },
            "retrieval_seed_terms": ["encouraging", "sign", "economy"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "not_falsifiable" in reason_codes

    def test_reject_claim_missing_subject_entities(self):
        """Claims without subject_entities should be rejected."""
        claim = {
            "claim_text": "Unemployment has increased significantly.",
            "subject_entities": [],  # Empty!
            "predicate_type": "measurement",
            "time_anchor": {"type": "relative", "value": "recently"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "dataset",
            },
            "retrieval_seed_terms": ["unemployment", "increased", "significantly"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "missing_subject_entities" in reason_codes

    def test_reject_claim_insufficient_seed_terms(self):
        """Claims with fewer than 3 retrieval_seed_terms should be rejected."""
        claim = {
            "claim_text": "Tesla stock rose 5%.",
            "subject_entities": ["Tesla"],
            "predicate_type": "measurement",
            "time_anchor": {"type": "explicit_date", "value": "2024-01-15"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "dataset",
            },
            "retrieval_seed_terms": ["Tesla", "stock"],  # Only 2!
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "insufficient_retrieval_seed_terms" in reason_codes

    def test_reject_claim_missing_falsifiability(self):
        """Claims without falsifiability object should be rejected."""
        claim = {
            "claim_text": "The company announced new products.",
            "subject_entities": ["company"],
            "predicate_type": "event",
            "time_anchor": {"type": "relative", "value": "today"},
            # No falsifiability!
            "retrieval_seed_terms": ["company", "announced", "products"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "missing_falsifiability" in reason_codes

    def test_reject_event_claim_with_unknown_time_anchor(self):
        """Event claims with unknown time anchor should be rejected."""
        claim = {
            "claim_text": "Google acquired a startup for $2 billion.",
            "subject_entities": ["Google"],
            "predicate_type": "event",  # Event requires time anchor
            "time_anchor": {"type": "unknown", "value": "unknown"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "reputable_news",
            },
            "retrieval_seed_terms": ["Google", "acquired", "startup", "billion"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "unknown_time_anchor" in reason_codes

    def test_reject_claim_empty_text(self):
        """Claims with empty text should be rejected."""
        claim = {
            "claim_text": "",
            "subject_entities": ["Something"],
            "predicate_type": "event",
            "time_anchor": {"type": "explicit_date", "value": "2024"},
            "falsifiability": {
                "is_falsifiable": True,
                "falsifiable_by": "reputable_news",
            },
            "retrieval_seed_terms": ["something", "here", "now"],
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert "empty_claim_text" in reason_codes

    def test_reject_multiple_issues(self):
        """Claims with multiple issues should report all reason codes."""
        claim = {
            "claim_text": "",  # Empty
            "subject_entities": [],  # Missing
            "predicate_type": "event",
            "time_anchor": {"type": "unknown", "value": "unknown"},  # Unknown for event
            # No falsifiability
            "retrieval_seed_terms": ["one"],  # Insufficient
        }
        ok, reason_codes = validate_core_claim(claim)
        assert ok is False
        assert len(reason_codes) >= 4
        assert "empty_claim_text" in reason_codes
        assert "missing_subject_entities" in reason_codes
        assert "missing_falsifiability" in reason_codes
        assert "insufficient_retrieval_seed_terms" in reason_codes


class TestExtractionStats:
    """Test ExtractionStats tracking."""

    def test_stats_tracking(self):
        stats = ExtractionStats()
        stats.claims_extracted_total = 10
        
        stats.record_emit()
        stats.record_emit()
        
        stats.record_drop(["not_falsifiable"])
        stats.record_drop(["not_falsifiable", "missing_subject_entities"])
        stats.record_drop(["unknown_time_anchor"])
        
        assert stats.claims_emitted_targets == 2
        assert stats.claims_dropped_nonverifiable == 3
        assert stats.drop_reason_counts["not_falsifiable"] == 2
        assert stats.drop_reason_counts["missing_subject_entities"] == 1
        assert stats.drop_reason_counts["unknown_time_anchor"] == 1

    def test_to_trace_dict(self):
        stats = ExtractionStats()
        stats.claims_extracted_total = 5
        stats.claims_dropped_nonverifiable = 2
        stats.claims_emitted_targets = 3
        stats.drop_reason_counts = {"not_falsifiable": 2}
        
        trace_dict = stats.to_trace_dict()
        
        assert trace_dict["claims_extracted_total"] == 5
        assert trace_dict["claims_dropped_nonverifiable"] == 2
        assert trace_dict["claims_emitted_targets"] == 3
        assert trace_dict["drop_reason_counts"]["not_falsifiable"] == 2


class TestTimeAnchorExemptPredicates:
    """Test that TIME_ANCHOR_EXEMPT_PREDICATES contains expected values."""

    def test_exempt_predicates_defined(self):
        assert "policy" in TIME_ANCHOR_EXEMPT_PREDICATES
        assert "ranking" in TIME_ANCHOR_EXEMPT_PREDICATES
        assert "existence" in TIME_ANCHOR_EXEMPT_PREDICATES
        assert "event" not in TIME_ANCHOR_EXEMPT_PREDICATES
        assert "measurement" not in TIME_ANCHOR_EXEMPT_PREDICATES
