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
Unit tests for M129 context anchoring and retrieval sanity gate.
"""

from spectrue_core.verification.search.search_escalation import (
    build_query_variants,
    _extract_context_entities,
)
from spectrue_core.verification.search.sanity_gate import (
    normalize_anchor_terms,
    check_sanity_gate,
)
from spectrue_core.agents.skills.coverage_skeleton import (
    compute_document_context_pool,
    _select_context_entities,
    CoverageSkeleton,
    SkeletonEvent,
    SkeletonMeasurement,
    SkeletonQuote,
)
from spectrue_core.schema.rgba_audit import RGBAStatus, RGBA_STATUS_LEGACY_CODES


class TestContextEntitiesInQueryVariants:
    """Tests for context entities in query generation."""

    def test_context_entities_included_in_query_variants_for_contextless_claim(self):
        """Claim with no subject_entities but context_entities → context in Q1/Q2."""
        claim = {
            "id": "c1",
            "subject_entities": [],
            "context_entities": ["kerosene", "todikamp"],
            "retrieval_seed_terms": ["treatment"],
        }
        variants = build_query_variants(claim)
        
        # Should have at least one variant
        assert len(variants) >= 1
        
        # At least one variant should contain context entity tokens
        combined_text = " ".join(v.text.lower() for v in variants)
        assert "kerosene" in combined_text or "todikamp" in combined_text

    def test_context_entities_included_with_subject_entities(self):
        """Claim with both subject_entities and context_entities → both in Q1/Q2."""
        claim = {
            "id": "c2",
            "subject_entities": ["walnut", "tincture"],
            "context_entities": ["kerosene", "todikamp"],
            "retrieval_seed_terms": ["treatment", "contraindicated"],
        }
        variants = build_query_variants(claim)
        
        assert len(variants) >= 1
        q1_text = variants[0].text.lower()
        
        # Should contain subject entities
        assert "walnut" in q1_text or "tincture" in q1_text
        # Should also contain context entities
        assert "kerosene" in q1_text or "todikamp" in q1_text

    def test_empty_query_blocked(self):
        """Claim with no queryable fields → returns empty list."""
        claim = {
            "id": "c3",
            "subject_entities": [],
            "context_entities": [],
            "retrieval_seed_terms": [],
        }
        variants = build_query_variants(claim)
        assert variants == []

    def test_extract_context_entities(self):
        """_extract_context_entities extracts from claim correctly."""
        claim = {
            "context_entities": ["Entity1", "Entity2", "Entity3"],
        }
        result = _extract_context_entities(claim, max_count=2)
        assert len(result) == 2
        assert result[0] == "Entity1"
        assert result[1] == "Entity2"


class TestSanityGate:
    """Tests for retrieval sanity gate."""

    def test_sanity_gate_flags_off_topic_sources(self):
        """Sources with no anchor term overlap → OFF_TOPIC."""
        anchor_terms = normalize_anchor_terms(
            subject_entities=["kerosene", "todikamp"],
            context_entities=["walnut", "tincture"],
            retrieval_seed_terms=["treatment", "contraindicated"],
        )
        
        # Off-topic sources (Nigeria oil crisis has no overlap)
        sources = [
            {
                "title": "Nigeria Oil Crisis",
                "snippet": "Reuters reports on Nigerian oil production decline affecting OPEC quotas.",
                "score": 0.6,
            },
        ]
        
        result = check_sanity_gate(sources, anchor_terms)
        assert result.decision == "off_topic"
        assert result.max_overlap_count == 0

    def test_sanity_gate_passes_on_topic_sources(self):
        """Sources with anchor term overlap → PASS."""
        anchor_terms = normalize_anchor_terms(
            subject_entities=["kerosene", "todikamp"],
            context_entities=["walnut", "tincture"],
            retrieval_seed_terms=["treatment"],
        )
        
        # On-topic sources (contains "kerosene" and "treatment")
        sources = [
            {
                "title": "Kerosene Treatment Warning",
                "snippet": "Health authorities warn about kerosene-based treatments.",
                "score": 0.7,
            },
        ]
        
        result = check_sanity_gate(sources, anchor_terms)
        assert result.decision == "pass"
        assert result.max_overlap_count >= 1

    def test_sanity_gate_empty_sources(self):
        """Empty sources → OFF_TOPIC with reason."""
        anchor_terms = normalize_anchor_terms(
            subject_entities=["test"],
            context_entities=[],
            retrieval_seed_terms=["search"],
        )
        
        result = check_sanity_gate([], anchor_terms)
        assert result.decision == "off_topic"
        assert "no_sources" in result.reasons

    def test_normalize_anchor_terms_deduplication(self):
        """Anchor terms should be deduplicated and normalized."""
        anchor_terms = normalize_anchor_terms(
            subject_entities=["Apple", "APPLE", "apple"],
            context_entities=["iPhone", "iphone"],
            retrieval_seed_terms=["sales", "Revenue"],
        )
        
        # Should contain normalized unique terms
        assert "apple" in anchor_terms
        assert "iphone" in anchor_terms
        assert "sales" in anchor_terms
        assert "revenue" in anchor_terms
        # Should not have duplicates
        assert len([t for t in anchor_terms if t == "apple"]) == 1


class TestDocumentContextPool:
    """Tests for document context pool computation."""

    def test_compute_document_context_pool_from_skeleton(self):
        """Pool includes entities from all skeleton item types."""
        skeleton = CoverageSkeleton(
            events=[
                SkeletonEvent(
                    id="evt_1",
                    subject_entities=["kerosene", "todikamp"],
                    verb_phrase="treats",
                    time_anchor=None,
                    location_anchor=None,
                    raw_span="Kerosene todikamp treats...",
                ),
            ],
            measurements=[
                SkeletonMeasurement(
                    id="msr_1",
                    subject_entities=["walnut"],
                    metric="concentration",
                    quantity_mentions=[],
                    time_anchor=None,
                    raw_span="Walnut concentration...",
                ),
            ],
            quotes=[
                SkeletonQuote(
                    id="qot_1",
                    speaker_entities=["Dr. Smith"],
                    quote_text="Treatment is dangerous",
                    raw_span="Dr. Smith said...",
                ),
            ],
            policies=[],
        )
        
        pool = compute_document_context_pool(skeleton)
        
        # Should include entities from events, measurements, quotes
        assert "kerosene" in pool
        assert "todikamp" in pool
        assert "walnut" in pool
        assert "Dr. Smith" in pool
        # Metric should also be included
        assert "concentration" in pool

    def test_select_context_entities_weak_claim(self):
        """Claim with < 2 entities gets context from pool."""
        claim_entities = ["only_one"]
        document_pool = ["entity1", "entity2", "entity3"]
        
        context, source = _select_context_entities(claim_entities, document_pool)
        
        assert source == "document_pool"
        assert len(context) > 0
        assert "only_one" not in context  # Should not duplicate

    def test_select_context_entities_rich_claim(self):
        """Claim with >= 2 entities doesn't need context."""
        claim_entities = ["entity1", "entity2"]
        document_pool = ["pool1", "pool2"]
        
        context, source = _select_context_entities(claim_entities, document_pool)
        
        assert source == "already_rich"
        assert context == []


class TestEvidenceMismatchStatus:
    """Tests for new EVIDENCE_MISMATCH status."""

    def test_evidence_mismatch_status_exists(self):
        """RGBAStatus should have EVIDENCE_MISMATCH."""
        assert hasattr(RGBAStatus, "EVIDENCE_MISMATCH")
        assert RGBAStatus.EVIDENCE_MISMATCH.value == "EVIDENCE_MISMATCH"

    def test_evidence_mismatch_legacy_code(self):
        """EVIDENCE_MISMATCH should map to -6."""
        assert RGBA_STATUS_LEGACY_CODES[RGBAStatus.EVIDENCE_MISMATCH] == -6

    def test_all_statuses_have_legacy_codes(self):
        """All RGBAStatus values should have legacy codes."""
        for status in RGBAStatus:
            assert status in RGBA_STATUS_LEGACY_CODES
