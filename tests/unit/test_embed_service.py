# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Unit tests for M109 EmbedService."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestEmbedServiceAvailability:
    """Test availability detection."""
    
    def test_import_works(self):
        """EmbedService can be imported."""
        from spectrue_core.embeddings import EmbedService
        assert EmbedService is not None
    
    def test_is_available_returns_bool(self):
        """is_available returns boolean."""
        from spectrue_core.embeddings import EmbedService
        result = EmbedService.is_available()
        assert isinstance(result, bool)


class TestEmbedServiceWithMock:
    """Test EmbedService with mocked sentence-transformers."""
    
    def test_embed_returns_array(self):
        """embed() returns numpy array when available."""
        from spectrue_core.embeddings.embed_service import EmbedService
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        with patch.object(EmbedService, '_model', mock_model):
            with patch.object(EmbedService, '_available', True):
                result = EmbedService.embed(["hello", "world"])
                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 3)
    
    def test_similarity_returns_float(self):
        """similarity() returns float in valid range."""
        from spectrue_core.embeddings.embed_service import EmbedService
        
        # Mock embed to return normalized vectors
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.707, 0.707, 0.0])  # 45 degrees
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([vec_a, vec_b])
        
        with patch.object(EmbedService, '_model', mock_model):
            with patch.object(EmbedService, '_available', True):
                result = EmbedService.similarity("hello", "world")
                assert isinstance(result, float)
                assert -1.0 <= result <= 1.0
    
    def test_similarity_empty_string_returns_zero(self):
        """similarity() with empty string returns 0."""
        from spectrue_core.embeddings.embed_service import EmbedService
        
        result = EmbedService.similarity("", "hello")
        assert result == 0.0
        
        result = EmbedService.similarity("hello", "")
        assert result == 0.0
    
    def test_batch_similarity_returns_list(self):
        """batch_similarity() returns list of floats."""
        from spectrue_core.embeddings.embed_service import EmbedService
        
        # Mock: query + 3 candidates = 4 vectors
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # query
            [0.9, 0.1, 0.0],  # candidate 1 - similar
            [0.0, 1.0, 0.0],  # candidate 2 - different
            [0.5, 0.5, 0.0],  # candidate 3 - medium
        ])
        
        with patch.object(EmbedService, '_model', mock_model):
            with patch.object(EmbedService, '_available', True):
                result = EmbedService.batch_similarity("query", ["a", "b", "c"])
                assert isinstance(result, list)
                assert len(result) == 3
                assert all(isinstance(x, float) for x in result)


class TestSplitSentences:
    """Test sentence splitter."""
    
    def test_splits_on_period(self):
        """Splits text on periods."""
        from spectrue_core.embeddings.embed_service import split_sentences
        
        text = "This is the first sentence here. This is the second longer sentence. And this is the third one too."
        result = split_sentences(text)
        assert len(result) == 3
    
    def test_filters_short_sentences(self):
        """Filters sentences shorter than 20 chars."""
        from spectrue_core.embeddings.embed_service import split_sentences
        
        text = "Hi. This is a longer sentence that should be kept."
        result = split_sentences(text)
        assert len(result) == 1
        assert "longer sentence" in result[0]
    
    def test_empty_text_returns_empty(self):
        """Empty text returns empty list."""
        from spectrue_core.embeddings.embed_service import split_sentences
        
        assert split_sentences("") == []
        assert split_sentences(None) == []


class TestExtractBestQuote:
    """Test quote extraction."""
    
    def test_returns_none_when_unavailable(self):
        """Returns None when embeddings unavailable."""
        from spectrue_core.embeddings.embed_service import extract_best_quote, EmbedService
        
        with patch.object(EmbedService, '_available', False):
            result = extract_best_quote("claim", "content")
            assert result is None
    
    def test_returns_content_when_no_sentences(self):
        """Returns truncated content when no splittable sentences."""
        from spectrue_core.embeddings.embed_service import extract_best_quote, EmbedService
        
        with patch.object(EmbedService, '_available', True):
            with patch('spectrue_core.embeddings.embed_service.split_sentences', return_value=[]):
                result = extract_best_quote("claim", "short")
                assert result == "short"
