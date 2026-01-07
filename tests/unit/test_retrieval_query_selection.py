# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for retrieval query selection and topic defaults.

Tests for the retrieval regression fix:
- Empty search_queries are repaired with keyword extraction
- normalized_text is never used as fallback for queries
- Default topic is "news" when search_method is missing/invalid
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from spectrue_core.agents.skills.claims import (
    sanitize_retrieval_response,
    extract_keywords_deterministic,
)
from spectrue_core.verification.orchestration.phase_runner import PhaseRunner


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
        
        with patch("spectrue_core.agents.skills.claims.Trace"):
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
        
        with patch("spectrue_core.agents.skills.claims.Trace"):
            result = sanitize_retrieval_response(data, claim_text=claim_text)
        
        assert result["search_queries"] == ["Ukraine offensive Kherson"]


class TestSelectClaimQueryNoNormalizedText:
    """Test that _select_claim_query never uses normalized_text."""

    def test_never_uses_normalized_text(self):
        """Query fallback uses text, NOT normalized_text."""
        search_mgr = MagicMock()
        runner = PhaseRunner(search_mgr, use_retrieval_loop=False)
        
        claim = {
            "id": "c1",
            "text": "Оригінальний український текст",
            "normalized_text": "English translation of the claim",
            "search_queries": [],
            "query_candidates": [],
        }
        
        with patch("spectrue_core.verification.orchestration.phase_runner.Trace"):
            query = runner._select_claim_query(claim)
        
        assert "English" not in query
        assert "translation" not in query
        assert "оригінальний" in query.lower() or "український" in query.lower()

    def test_prefers_search_queries(self):
        """When search_queries exist, use them."""
        search_mgr = MagicMock()
        runner = PhaseRunner(search_mgr, use_retrieval_loop=False)
        
        claim = {
            "id": "c1",
            "text": "Some claim",
            "search_queries": ["Explicit query from planner"],
        }
        
        with patch("spectrue_core.verification.orchestration.phase_runner.Trace"):
            query = runner._select_claim_query(claim)
        
        assert query == "Explicit query from planner"


class TestDefaultTopicIsNews:
    """Test that default topic is 'news' when search_method missing/invalid."""

    @pytest.mark.asyncio
    async def test_default_topic_news_when_method_missing(self):
        """Topic defaults to 'news' when search_method is missing."""
        search_mgr = MagicMock()
        search_mgr.search_phase = AsyncMock(return_value=(None, []))
        
        runner = PhaseRunner(search_mgr, use_retrieval_loop=False)
        
        claim = {
            "id": "c1",
            "text": "Test claim",
            "search_queries": ["test query"],
        }
        
        with patch("spectrue_core.verification.orchestration.phase_runner.Trace"):
            await runner._search_by_phase(claim, phase=None, query_override="test")
        
        call_args = search_mgr.search_phase.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_topic_selection_ignores_search_method(self):
        """Topic selection relies on structured claim fields, ignoring legacy search_method."""
        search_mgr = MagicMock()
        search_mgr.search_phase = AsyncMock(return_value=(None, []))
        
        runner = PhaseRunner(search_mgr, use_retrieval_loop=False)
        
        claim = {
            "id": "c1",
            "text": "Test claim",
            "search_queries": ["test query"],
            "search_method": "general_search",
        }
        
        with patch("spectrue_core.verification.orchestration.phase_runner.Trace"):
            await runner._search_by_phase(claim, phase=None, query_override="test")
        
        call_args = search_mgr.search_phase.call_args
        _, kwargs = call_args
        assert kwargs.get("topic") == "news"


class TestRegressionGoldenScenario:
    """Golden test for the regression scenario."""

    def test_empty_queries_normalized_text_scenario(self):
        """
        Regression scenario: empty queries should use keywords from text,
        NOT normalized_text.
        """
        search_mgr = MagicMock()
        runner = PhaseRunner(search_mgr, use_retrieval_loop=False)
        
        claim = {
            "id": "c1",
            "text": "Землетрус зруйнував будівлю в Києві",
            "normalized_text": "The earthquake destroyed the building in Kyiv",
            "search_queries": [],
            "query_candidates": [],
        }
        
        with patch("spectrue_core.verification.orchestration.phase_runner.Trace"):
            query = runner._select_claim_query(claim)
        
        assert "earthquake" not in query.lower()
        assert "destroyed" not in query.lower()
        assert any(w in query.lower() for w in ["землетрус", "зруйнував", "будівлю", "києві"])
