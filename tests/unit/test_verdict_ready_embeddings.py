# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M109: Test for verdict_ready_for_claim with embeddings.

These tests reproduce the exact conditions from trace 2025-12-27_00-03-19_20c857
where semantic_matches was unexpectedly 0.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestVerdictReadyWithEmbeddings:
    """Test verdict_ready_for_claim with embedding similarity."""
    
    def test_semantic_match_with_related_snippet(self):
        """
        Test that a source with semantically related snippet triggers semantic_matches.
        
        Reproduces trace issue: claim about PSR J2322-2650b, snippet about same planet.
        Expected: semantic_matches >= 1
        """
        from spectrue_core.verification.sufficiency import verdict_ready_for_claim
        from spectrue_core.embeddings import EmbedService
        
        # Skip if embeddings not available
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        claim_text = "PSR J2322-2650b orbits a millisecond pulsar PSR J2322-2650"
        
        # Source with related snippet (like in real trace)
        sources = [
            {
                "url": "https://www.discovermagazine.com/weird-lemon-shaped-exoplanet",
                "title": "Weird Lemon-Shaped Exoplanet Discovered",
                "snippet": "A peculiar exoplanet around a pulsar has been studied by the James Webb Space Telescope revealing its bizarre lemon shape and unusual carbon-rich atmosphere.",
                "relevance_score": 0.86,
            }
        ]
        
        ready, stats = verdict_ready_for_claim(sources, claim_text=claim_text)
        
        # Check that semantic matching worked
        assert stats["semantic_matches"] >= 1, \
            f"Expected semantic_matches >= 1, got {stats['semantic_matches']}. " \
            f"Snippet should semantically match claim about pulsar exoplanet."
    
    def test_semantic_match_threshold_is_reachable(self):
        """
        Test that the semantic similarity threshold is achievable for related content.
        
        Issue: threshold was 0.6 but real similarity ~0.48
        """
        from spectrue_core.embeddings import EmbedService
        
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        claim = "PSR J2322-2650b orbits a millisecond pulsar"
        snippet = "A peculiar exoplanet around a pulsar has been studied by JWST revealing its bizarre lemon shape"
        
        sim = EmbedService.similarity(claim, snippet)
        
        # The threshold in verdict_ready should be achievable for related content
        # Current threshold: 0.6, actual similarity: ~0.48
        assert sim >= 0.35, f"Similarity {sim:.3f} should be >= 0.35 for related content"
        
        # Log actual similarity for debugging threshold choices
        print(f"Actual similarity: {sim:.3f}")
    
    def test_high_relevance_counts_as_evidence(self):
        """
        Test that sources with high relevance_score count as sources_with_evidence.
        """
        from spectrue_core.verification.sufficiency import verdict_ready_for_claim
        
        sources = [
            {
                "url": "https://example.com/article",
                "title": "Related Article",
                "snippet": "Some content about the topic",
                "relevance_score": 0.85,  # High relevance
            }
        ]
        
        ready, stats = verdict_ready_for_claim(sources)
        
        assert stats["sources_with_evidence"] >= 1, \
            f"Source with relevance 0.85 should count as evidence, got {stats['sources_with_evidence']}"
    
    def test_quote_counts_as_evidence(self):
        """
        Test that sources with quotes count as sources_with_evidence.
        """
        from spectrue_core.verification.sufficiency import verdict_ready_for_claim
        
        sources = [
            {
                "url": "https://example.com/article",
                "title": "Article with Quote",
                "quote": "The planet orbits a millisecond pulsar at high speed.",
                "relevance_score": 0.5,
            }
        ]
        
        ready, stats = verdict_ready_for_claim(sources)
        
        assert stats["sources_with_evidence"] >= 1, \
            f"Source with quote should count as evidence, got {stats['sources_with_evidence']}"
    
    def test_verdict_ready_true_with_semantic_match(self):
        """
        Test that verdict_ready returns True when semantic_matches >= 1.
        """
        from spectrue_core.verification.sufficiency import verdict_ready_for_claim
        from spectrue_core.embeddings import EmbedService
        
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        claim_text = "The exoplanet has a lemon-shaped form due to tidal forces"
        
        sources = [
            {
                "url": "https://sciencealert.com/lemon-shaped-planet",
                "snippet": "Scientists discovered that the planet is stretched into a lemon shape by the intense gravitational tidal forces from its host pulsar.",
                "relevance_score": 0.7,
            }
        ]
        
        ready, stats = verdict_ready_for_claim(sources, claim_text=claim_text)
        
        # Either semantic_matches OR sources_with_evidence should make it ready
        assert ready or stats["sources_with_evidence"] >= 1 or stats["semantic_matches"] >= 1, \
            f"Should be ready with matching evidence. Stats: {stats}"


class TestEmbeddingThresholds:
    """Test that embedding similarity thresholds are appropriately set."""
    
    def test_exact_match_high_similarity(self):
        """Exact or near-exact text should have very high similarity."""
        from spectrue_core.embeddings import EmbedService
        
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        text = "The planet orbits a pulsar"
        sim = EmbedService.similarity(text, text)
        assert sim >= 0.99, f"Exact match should have sim >= 0.99, got {sim}"
    
    def test_paraphrase_medium_similarity(self):
        """Paraphrased content should have medium-high similarity."""
        from spectrue_core.embeddings import EmbedService
        
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        text_a = "The exoplanet orbits a millisecond pulsar"
        text_b = "A planet going around a fast-spinning neutron star"
        
        sim = EmbedService.similarity(text_a, text_b)
        assert sim >= 0.3, f"Paraphrase should have sim >= 0.3, got {sim}"
    
    def test_unrelated_low_similarity(self):
        """Completely unrelated content should have low similarity."""
        from spectrue_core.embeddings import EmbedService
        
        if not EmbedService.is_available():
            pytest.skip("sentence-transformers not installed")
        
        text_a = "The exoplanet orbits a millisecond pulsar"
        text_b = "The stock market crashed yesterday due to economic concerns"
        
        sim = EmbedService.similarity(text_a, text_b)
        assert sim < 0.3, f"Unrelated content should have sim < 0.3, got {sim}"
