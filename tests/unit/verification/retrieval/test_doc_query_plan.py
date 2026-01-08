# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for CEGS-MVP document query planning (M130).

Tests cover:
- Entity collection with context_entities and subject_entities
- Deterministic query generation (2-4 queries)
- Empty entity handling
- Query length bounds (<=80 chars)
- Module wiring verification
"""

from spectrue_core.verification.retrieval.cegs_mvp import (
    build_doc_query_plan,
    collect_document_entities,
    collect_seed_terms,
    _extract_entities_from_claim,
)
from spectrue_core.verification.claims.coverage_anchors import (
    Anchor,
    AnchorKind,
)


class TestCollectDocumentEntities:
    """Tests for collect_document_entities function (B)."""

    def test_collects_from_context_entities(self):
        """Entities from context_entities are collected."""
        claims = [
            {"context_entities": ["Apple", "iPhone"], "subject_entities": []},
            {"context_entities": ["Apple", "MacBook"], "subject_entities": []},
        ]
        entities = collect_document_entities(claims)
        
        assert "Apple" in entities
        assert len(entities) >= 2

    def test_collects_from_subject_entities(self):
        """Entities from subject_entities are collected."""
        claims = [
            {"context_entities": [], "subject_entities": ["Tesla", "Elon Musk"]},
        ]
        entities = collect_document_entities(claims)
        
        assert "Tesla" in entities
        assert "Elon Musk" in entities

    def test_deduplication_preserves_first_casing(self):
        """Duplicate entities are deduplicated, first-seen casing preserved."""
        claims = [
            {"context_entities": ["APPLE", "apple", "Apple"], "subject_entities": []},
        ]
        entities = collect_document_entities(claims)
        
        # Should have only one "apple" variant
        apple_variants = [e for e in entities if e.lower() == "apple"]
        assert len(apple_variants) == 1
        assert apple_variants[0] == "APPLE"  # First-seen casing

    def test_filters_short_entities(self):
        """Entities with len < 2 are filtered out."""
        claims = [
            {"context_entities": ["a", "ab", "abc", "abcd"], "subject_entities": []},
        ]
        entities = collect_document_entities(claims)
        
        assert "a" not in entities
        assert "ab" in entities
        assert "abc" in entities

    def test_frequency_ranking(self):
        """Most frequent entities are ranked first."""
        claims = [
            {"context_entities": ["rare"], "subject_entities": []},
            {"context_entities": ["common", "common"], "subject_entities": []},
            {"context_entities": ["common"], "subject_entities": []},
        ]
        entities = collect_document_entities(claims)
        
        # "common" appears 3 times, "rare" appears 1 time
        assert entities[0].lower() == "common"

    def test_max_count_limit(self):
        """Returns at most max_count entities."""
        claims = [
            {"context_entities": [f"entity_{i}" for i in range(20)], "subject_entities": []},
        ]
        entities = collect_document_entities(claims, max_count=5)
        
        assert len(entities) == 5

    def test_empty_claims_returns_empty(self):
        """Empty claims list returns empty entities."""
        entities = collect_document_entities([])
        assert entities == []

    def test_claims_without_entities_returns_empty(self):
        """Claims without entity fields return empty."""
        claims = [
            {"text": "Some claim without entities"},
            {"normalized_text": "Another claim"},
        ]
        entities = collect_document_entities(claims)
        assert entities == []


class TestCollectSeedTerms:
    """Tests for collect_seed_terms function."""

    def test_collects_seed_terms(self):
        """Seed terms are collected from claims."""
        claims = [
            {"retrieval_seed_terms": ["treatment", "safety", "efficacy"]},
            {"retrieval_seed_terms": ["treatment", "dosage"]},
        ]
        terms = collect_seed_terms(claims)
        
        assert "treatment" in terms
        assert len(terms) >= 2

    def test_frequency_ranking_seed_terms(self):
        """Most frequent seed terms are ranked first."""
        claims = [
            {"retrieval_seed_terms": ["common", "rare"]},
            {"retrieval_seed_terms": ["common", "common"]},
        ]
        terms = collect_seed_terms(claims)
        
        assert terms[0].lower() == "common"


class TestBuildDocQueryPlan:
    """Tests for build_doc_query_plan function (C)."""

    def test_produces_queries_with_context_entities(self):
        """Claims with context_entities produce non-empty queries."""
        claims = [
            {"context_entities": ["kerosene", "todikamp", "treatment"]},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) >= 2
        assert all(isinstance(q, str) for q in queries)

    def test_produces_queries_without_subject_entities(self):
        """Claims with only context_entities (no subject_entities) still work."""
        claims = [
            {"context_entities": ["entity1", "entity2", "entity3"], "subject_entities": []},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) >= 2

    def test_produces_queries_with_only_subject_entities(self):
        """Claims with only subject_entities (no context_entities) work."""
        claims = [
            {"subject_entities": ["Tesla", "SpaceX", "Elon Musk"], "context_entities": []},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) >= 2

    def test_returns_empty_when_no_entities(self):
        """Returns empty list when no entities found."""
        claims = [
            {"text": "A claim without entities"},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert queries == []

    def test_includes_date_anchor_in_query(self):
        """Date anchors are included in Q1."""
        claims = [
            {"context_entities": ["event", "conference", "announcement"]},
        ]
        anchors = [
            Anchor(
                anchor_id="t1",
                kind=AnchorKind.TIME,
                span_text="2024-01-15",
                char_start=0,
                char_end=10,
                context_window="on 2024-01-15",
            ),
        ]
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) >= 2
        # At least one query should contain the date
        assert any("2024-01-15" in q for q in queries)

    def test_includes_numeric_anchor_in_query(self):
        """Numeric anchors are included in Q2."""
        claims = [
            {"context_entities": ["revenue", "company", "growth"]},
        ]
        anchors = [
            Anchor(
                anchor_id="n1",
                kind=AnchorKind.NUMBER,
                span_text="$5 billion",
                char_start=0,
                char_end=10,
                context_window="raised $5 billion",
            ),
        ]
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) >= 2
        # At least one query should contain the number
        assert any("$5 billion" in q for q in queries)

    def test_query_max_length_80_chars(self):
        """All queries are <= 80 characters."""
        claims = [
            {"context_entities": ["very_long_entity_name_" + str(i) for i in range(10)]},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        for q in queries:
            assert len(q) <= 80, f"Query exceeds 80 chars: {q}"

    def test_max_4_queries(self):
        """Returns at most 4 queries."""
        claims = [
            {"context_entities": [f"entity_{i}" for i in range(20)]},
            {"retrieval_seed_terms": [f"term_{i}" for i in range(10)]},
        ]
        # Add both time and numeric anchors
        anchors = [
            Anchor(anchor_id="t1", kind=AnchorKind.TIME, span_text="2024", 
                   char_start=0, char_end=4, context_window="in 2024"),
            Anchor(anchor_id="n1", kind=AnchorKind.NUMBER, span_text="100%",
                   char_start=0, char_end=4, context_window="grew 100%"),
        ]
        
        queries = build_doc_query_plan(claims, anchors)
        
        assert len(queries) <= 4

    def test_queries_are_unique(self):
        """No duplicate queries are returned."""
        claims = [
            {"context_entities": ["same", "entity"]},
        ]
        anchors = []
        
        queries = build_doc_query_plan(claims, anchors)
        
        # Check for case-insensitive uniqueness
        lower_queries = [q.lower() for q in queries]
        assert len(lower_queries) == len(set(lower_queries))


class TestModuleWiring:
    """Tests for module wiring verification (D)."""

    def test_module_path_correct(self):
        """Verify correct module is imported."""
        import spectrue_core.verification.retrieval.cegs_mvp as cegs
        
        assert "spectrue_core" in cegs.__name__
        assert "cegs_mvp" in cegs.__file__
        assert hasattr(cegs, "build_doc_query_plan")
        assert hasattr(cegs, "collect_document_entities")


class TestExtractEntitiesFromClaim:
    """Tests for _extract_entities_from_claim helper."""

    def test_extracts_context_entities(self):
        """Extracts from context_entities field."""
        claim = {"context_entities": ["A", "B"]}
        entities = _extract_entities_from_claim(claim)
        
        assert "A" in entities
        assert "B" in entities

    def test_extracts_subject_entities(self):
        """Extracts from subject_entities field."""
        claim = {"subject_entities": ["X", "Y"]}
        entities = _extract_entities_from_claim(claim)
        
        assert "X" in entities
        assert "Y" in entities

    def test_extracts_subject_field(self):
        """Extracts from subject field."""
        claim = {"subject": "MainSubject"}
        entities = _extract_entities_from_claim(claim)
        
        assert "MainSubject" in entities

    def test_handles_missing_fields(self):
        """Handles claims with missing entity fields."""
        claim = {"text": "Just text, no entities"}
        entities = _extract_entities_from_claim(claim)
        
        assert entities == []
