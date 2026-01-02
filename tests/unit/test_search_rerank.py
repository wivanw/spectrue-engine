# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for search result reranking (Phase 3).

Tests that reranking sorts by combined score without discarding results.
"""

import pytest
from spectrue_core.verification.search.search_policy import (
    rerank_search_results,
    filter_search_results,
)


class TestRerankSearchResults:
    """Test the reranking function."""
    
    def test_rerank_sorts_by_combined_score(self):
        """Results should be sorted by λ·provider + (1-λ)·relevance."""
        results = [
            {"url": "a.com", "score": 0.5, "relevance_score": 0.5},  # combined = 0.5
            {"url": "b.com", "score": 0.9, "relevance_score": 0.3},  # combined = 0.7*0.9 + 0.3*0.3 = 0.72
            {"url": "c.com", "score": 0.3, "relevance_score": 0.9},  # combined = 0.7*0.3 + 0.3*0.9 = 0.48
        ]
        
        reranked = rerank_search_results(results, rerank_lambda=0.7)
        
        # b.com should be first (highest combined score)
        assert reranked[0]["url"] == "b.com"
        # All results preserved
        assert len(reranked) == 3
    
    def test_rerank_no_results_discarded(self):
        """Even low-score results should be kept (just ranked lower)."""
        results = [
            {"url": "high.com", "score": 0.9, "relevance_score": 0.9},
            {"url": "low.com", "score": 0.1, "relevance_score": 0.1},
        ]
        
        reranked = rerank_search_results(results)
        
        # Both results preserved
        assert len(reranked) == 2
        assert any(r["url"] == "low.com" for r in reranked)
    
    def test_rerank_top_k_limits_output(self):
        """top_k should limit number of results returned."""
        results = [
            {"url": f"{i}.com", "score": 0.5, "relevance_score": 0.5}
            for i in range(10)
        ]
        
        reranked = rerank_search_results(results, top_k=3)
        
        assert len(reranked) == 3
    
    def test_rerank_skips_bad_extensions(self):
        """Files with skip_extensions should be removed."""
        results = [
            {"url": "good.com", "score": 0.8, "relevance_score": 0.8},
            {"url": "file.txt", "score": 0.9, "relevance_score": 0.9},
            {"url": "data.xml", "score": 0.9, "relevance_score": 0.9},
        ]
        
        reranked = rerank_search_results(results)
        
        assert len(reranked) == 1
        assert reranked[0]["url"] == "good.com"
    
    def test_rerank_handles_missing_scores(self):
        """Missing scores should use defaults."""
        results = [
            {"url": "no_relevance.com", "score": 0.8},
            {"url": "no_provider.com", "relevance_score": 0.7},
            {"url": "no_scores.com"},
        ]
        
        reranked = rerank_search_results(results)
        
        # Should not crash
        assert len(reranked) == 3
        # All should have _rerank_score
        for r in reranked:
            assert "_rerank_score" in r
    
    def test_rerank_lambda_affects_ordering(self):
        """Different lambda values should change result order."""
        results = [
            {"url": "provider_high.com", "score": 0.9, "relevance_score": 0.1},
            {"url": "relevance_high.com", "score": 0.1, "relevance_score": 0.9},
        ]
        
        # With λ=0.9 (favor provider score)
        reranked_provider = rerank_search_results(results, rerank_lambda=0.9)
        assert reranked_provider[0]["url"] == "provider_high.com"
        
        # With λ=0.1 (favor relevance score)
        reranked_relevance = rerank_search_results(results, rerank_lambda=0.1)
        assert reranked_relevance[0]["url"] == "relevance_high.com"
    
    def test_rerank_empty_input(self):
        """Empty input should return empty output."""
        assert rerank_search_results([]) == []
        assert rerank_search_results(None) == []


class TestFilterVsRerank:
    """Compare filter (old) vs rerank (new) behavior."""
    
    def test_filter_discards_low_relevance(self):
        """Old filter removes results below threshold."""
        results = [
            {"url": "high.com", "relevance_score": 0.5},
            {"url": "low.com", "relevance_score": 0.1},  # Below 0.15 threshold
        ]
        
        filtered = filter_search_results(results, min_relevance_score=0.15)
        
        # Low result is discarded
        assert len(filtered) == 1
        assert filtered[0]["url"] == "high.com"
    
    def test_rerank_keeps_low_relevance(self):
        """New rerank keeps all results."""
        results = [
            {"url": "high.com", "score": 0.5, "relevance_score": 0.5},
            {"url": "low.com", "score": 0.5, "relevance_score": 0.1},
        ]
        
        reranked = rerank_search_results(results)
        
        # Both results kept
        assert len(reranked) == 2
        # Low result is just ranked lower
        assert reranked[-1]["url"] == "low.com"

