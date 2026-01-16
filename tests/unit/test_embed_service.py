# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Unit tests for EmbedService."""

from unittest.mock import patch, MagicMock


class TestEmbedServiceAvailability:
    """Test availability detection."""
    
    def test_import_works(self):
        """EmbedService can be imported."""
        from spectrue_core.utils.embedding_service import EmbedService
        assert EmbedService is not None
    
    def test_is_available_returns_bool(self):
        """is_available returns boolean."""
        from spectrue_core.utils.embedding_service import EmbedService
        result = EmbedService.is_available()
        assert isinstance(result, bool)


class TestEmbedServiceWithMock:
    """Test EmbedService with mocked OpenAI embeddings."""
    
    def test_embed_returns_list(self):
        """embed() returns list of vectors when available."""
        from spectrue_core.utils.embedding_service import EmbedService
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(EmbedService, "_client", mock_client):
            EmbedService._cache = {}
            result = EmbedService.embed(["hello", "world"])
            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(row, list) for row in result)
            assert all(len(row) == 3 for row in result)
    
    def test_similarity_returns_float(self):
        """similarity() returns float in valid range."""
        from spectrue_core.utils.embedding_service import EmbedService
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.707, 0.707, 0.0]  # 45 degrees

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=vec_a),
            MagicMock(embedding=vec_b),
        ]
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(EmbedService, "_client", mock_client):
            EmbedService._cache = {}
            result = EmbedService.similarity("hello", "world")
            assert isinstance(result, float)
            assert -1.0 <= result <= 1.0
    
    def test_similarity_empty_string_returns_zero(self):
        """similarity() with empty string returns 0."""
        from spectrue_core.utils.embedding_service import EmbedService
        
        result = EmbedService.similarity("", "hello")
        assert result == 0.0
        
        result = EmbedService.similarity("hello", "")
        assert result == 0.0
    
    def test_batch_similarity_returns_list(self):
        """batch_similarity() returns list of floats."""
        from spectrue_core.utils.embedding_service import EmbedService
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[1.0, 0.0, 0.0]),  # query
            MagicMock(embedding=[0.9, 0.1, 0.0]),  # candidate 1 - similar
            MagicMock(embedding=[0.0, 1.0, 0.0]),  # candidate 2 - different
            MagicMock(embedding=[0.5, 0.5, 0.0]),  # candidate 3 - medium
        ]
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(EmbedService, "_client", mock_client):
            EmbedService._cache = {}
            result = EmbedService.batch_similarity("query", ["a", "b", "c"])
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(x, float) for x in result)


class TestSplitSentences:
    """Test sentence splitter."""
    
    def test_splits_on_period(self):
        """Splits text on periods."""
        from spectrue_core.utils.embedding_service import split_sentences
        
        text = "This is the first sentence here. This is the second longer sentence. And this is the third one too."
        result = split_sentences(text)
        assert len(result) == 3
    
    def test_filters_short_sentences(self):
        """Filters sentences shorter than 20 chars."""
        from spectrue_core.utils.embedding_service import split_sentences
        
        text = "Hi. This is a longer sentence that should be kept."
        result = split_sentences(text)
        assert len(result) == 1
        assert "longer sentence" in result[0]
    
    def test_empty_text_returns_empty(self):
        """Empty text returns empty list."""
        from spectrue_core.utils.embedding_service import split_sentences
        
        assert split_sentences("") == []
        assert split_sentences(None) == []


class TestExtractBestQuote:
    """Test quote extraction."""
    
    def test_returns_none_when_unavailable(self):
        """Returns None when embeddings unavailable."""
        from spectrue_core.utils.embedding_service import extract_best_quote, EmbedService
        
        with patch.object(EmbedService, "is_available", return_value=False):
            result = extract_best_quote("claim", "content")
            assert result is None
    
    def test_returns_content_when_no_sentences(self):
        """Returns truncated content when no splittable sentences."""
        from spectrue_core.utils.embedding_service import extract_best_quote, EmbedService
        
        with patch.object(EmbedService, "is_available", return_value=True):
            with patch('spectrue_core.utils.embedding_service.split_sentences', return_value=[]):
                result = extract_best_quote("claim", "short")
                assert result == "short"
