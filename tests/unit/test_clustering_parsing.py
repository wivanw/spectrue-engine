"""
Unit tests for clustering_parsing.py — specifically build_claims_lite.

These tests ensure that search_query is correctly extracted from claims
to help LLM match sources to claims during stance clustering.
"""
from spectrue_core.agents.skills.clustering_parsing import build_claims_lite


class TestBuildClaimsLite:
    """Tests for build_claims_lite function."""

    def test_extracts_search_query_from_query_candidates(self):
        """M105: search_query should be extracted from query_candidates[0]['text']."""
        claims = [
            {
                "id": "c1",
                "text": "The universe is 13.8 billion years old.",
                "query_candidates": [
                    {"text": "age of universe billion years", "locale": "en"},
                    {"text": "how old is the universe", "locale": "en"},
                ],
            }
        ]
        result = build_claims_lite(claims)
        
        assert len(result) == 1
        assert "search_query" in result[0]
        # query_candidates are dicts with "text" key, should extract text
        assert result[0]["search_query"] == "age of universe billion years"

    def test_extracts_search_query_from_search_queries(self):
        """M105: search_query should be extracted from search_queries array."""
        claims = [
            {
                "id": "c1",
                "text": "Vaccination rates declined.",
                "search_queries": ["vaccination rates decline 2023", "vaccine uptake statistics"],
            }
        ]
        result = build_claims_lite(claims)
        
        assert len(result) == 1
        assert result[0]["search_query"] == "vaccination rates decline 2023"

    def test_fallback_to_text_when_no_queries(self):
        """When no query fields, should fallback to text[:100]."""
        claims = [
            {
                "id": "c1",
                "text": "This is a claim without any search queries defined.",
            }
        ]
        result = build_claims_lite(claims)
        
        assert len(result) == 1
        assert result[0]["search_query"] == "This is a claim without any search queries defined."

    def test_prefers_search_queries_over_query_candidates(self):
        """search_queries should take priority over query_candidates."""
        claims = [
            {
                "id": "c1",
                "text": "Some claim.",
                "search_queries": ["preferred query"],
                "query_candidates": [{"text": "fallback query", "locale": "en"}],
            }
        ]
        result = build_claims_lite(claims)
        
        assert result[0]["search_query"] == "preferred query"

    def test_handles_empty_query_arrays(self):
        """Empty arrays should fallback to text."""
        claims = [
            {
                "id": "c1",
                "text": "Claim with empty arrays.",
                "search_queries": [],
                "query_candidates": [],
            }
        ]
        result = build_claims_lite(claims)
        
        assert result[0]["search_query"] == "Claim with empty arrays."

    def test_includes_all_required_fields(self):
        """Result should include id, text, assertions, search_query."""
        claims = [
            {
                "id": "c1",
                "text": "Test claim.",
                "search_queries": ["test query"],
            }
        ]
        result = build_claims_lite(claims)
        
        assert "id" in result[0]
        assert "text" in result[0]
        assert "assertions" in result[0]
        assert "search_query" in result[0]
