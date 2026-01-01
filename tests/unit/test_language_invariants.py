# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for language consistency invariants (Phase 4).

Tests that search queries use original claim.text language,
and that normalized_text is not used for search.
"""

import pytest
from spectrue_core.agents.skills.clustering_parsing import _extract_search_query


class TestExtractSearchQuery:
    """Test search query extraction uses correct text field."""
    
    def test_prefers_search_queries_over_text(self):
        """If search_queries exist, use them (not text)."""
        claim = {
            "text": "Текст українською",
            "normalized_text": "Translated to English",
            "search_queries": ["Ukrainian query"],
        }
        result = _extract_search_query(claim)
        assert result == "Ukrainian query"
    
    def test_prefers_query_candidates_over_text(self):
        """If query_candidates exist, use them."""
        claim = {
            "text": "Текст українською",
            "normalized_text": "Translated to English",
            "query_candidates": [{"text": "Query from candidate"}],
        }
        result = _extract_search_query(claim)
        assert result == "Query from candidate"
    
    def test_fallback_uses_text_not_normalized(self):
        """Fallback should use text, not normalized_text."""
        claim = {
            "text": "Оригінальний текст",
            "normalized_text": "Translated text",
        }
        result = _extract_search_query(claim)
        # INVARIANT: should use original text for search
        assert result == "Оригінальний текст"
        assert "Translated" not in result
    
    def test_dict_claim_uses_text(self):
        """Dict claim without attributes uses text field."""
        claim = {
            "text": "Original claim text",
        }
        result = _extract_search_query(claim)
        assert result == "Original claim text"
    
    def test_empty_claim_returns_empty(self):
        """Empty claim returns empty string."""
        result = _extract_search_query({})
        assert result == ""


class TestLanguageInvariant:
    """Test language consistency invariants."""
    
    def test_ukrainian_claim_keeps_ukrainian_query(self):
        """Ukrainian claim should produce Ukrainian search query."""
        claim = {
            "text": "Українські війська звільнили місто",
            "normalized_text": "Ukrainian forces liberated the city",
            # No search_queries - triggers fallback
        }
        query = _extract_search_query(claim)
        # Query should be in Ukrainian (original)
        assert "Українські" in query or "українські" in query.lower()
        assert "Ukrainian" not in query
    
    def test_english_claim_keeps_english_query(self):
        """English claim should produce English search query."""
        claim = {
            "text": "Tesla announced new model",
            "normalized_text": "Tesla announced new model",  # Same for English
        }
        query = _extract_search_query(claim)
        assert "Tesla" in query

