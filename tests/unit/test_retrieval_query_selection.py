# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for retrieval query repair.

Tests for the retrieval regression fix:
- Empty search_queries are repaired with keyword extraction
"""

from unittest.mock import patch

from spectrue_core.adapters.llm.claims import (
    sanitize_retrieval_response,
    extract_keywords_deterministic,
)


class TestKeywordExtraction:
    """Test deterministic keyword extraction helper."""

    def test_extracts_keywords_from_text(self):
        """Basic keyword extraction works."""
        text = "Ukrainian forces liberated the city of Kherson"
        result = extract_keywords_deterministic(text)
        assert "ukrainian" in result.lower()
        assert "forces" in result.lower()
        assert "." not in result

    def test_drops_short_tokens(self):
        """Tokens with length < 3 are dropped."""
        text = "A is an example of a thing"
        result = extract_keywords_deterministic(text)
        assert "example" in result.lower()
        assert " a " not in f" {result.lower()} "

    def test_handles_unicode_text(self):
        """Works with Ukrainian/Cyrillic text."""
        text = "Українські війська звільнили місто"
        result = extract_keywords_deterministic(text)
        assert "українські" in result.lower()

    def test_unique_tokens_only(self):
        """Duplicate tokens are removed."""
        text = "test test test different word word"
        result = extract_keywords_deterministic(text)
        count = result.lower().split().count("test")
        assert count == 1

    def test_max_tokens_limit(self):
        """Respects max_tokens parameter."""
        text = "one two three four five six seven eight nine ten"
        result = extract_keywords_deterministic(text, max_tokens=3)
        tokens = result.split()
        assert len(tokens) <= 3


class TestSanitizeRetrievalResponse:
    """Test sanitize_retrieval_response with query repair."""

    def test_empty_queries_repaired_with_keywords(self):
        """Empty search_queries are repaired using keyword extraction."""
        data = {
            "claim_category": "FACTUAL",
            "harm_potential": 3,
            "verification_target": "reality",
            "claim_role": "core",
            "search_queries": [],
        }
        claim_text = "Українські війська звільнили місто Херсон"
        
        with patch("spectrue_core.adapters.llm.claims.Trace"):
            result = sanitize_retrieval_response(data, claim_text=claim_text)
        
        assert "search_queries" in result
        assert len(result["search_queries"]) > 0
        assert "українські" in result["search_queries"][0].lower()

    def test_valid_queries_not_modified(self):
        """Existing valid queries are preserved."""
        data = {
            "claim_category": "FACTUAL",
            "search_queries": ["Ukraine offensive Kherson"],
        }
        claim_text = "Different claim text"
        
        with patch("spectrue_core.adapters.llm.claims.Trace"):
            result = sanitize_retrieval_response(data, claim_text=claim_text)
        
        assert result["search_queries"] == ["Ukraine offensive Kherson"]
