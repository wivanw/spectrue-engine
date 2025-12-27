# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M109: Local Embedding Service

Provides semantic similarity using sentence-transformers.
Model: all-MiniLM-L6-v2 (~23MB, ~50ms per embed)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model - small, fast, good quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbedService:
    """
    Singleton embedding service using sentence-transformers.
    
    Lazy loads the model on first use to avoid startup cost.
    Thread-safe for inference (model is read-only after load).
    """
    
    _model: "SentenceTransformer | None" = None
    _model_name: str = DEFAULT_MODEL
    _available: bool | None = None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers is installed."""
        if cls._available is None:
            try:
                import sentence_transformers  # noqa: F401
                cls._available = True
            except ImportError:
                cls._available = False
                logger.debug("[Embeddings] sentence-transformers not installed")
        return cls._available
    
    @classmethod
    def get_model(cls) -> "SentenceTransformer":
        """Get or load the embedding model (lazy singleton)."""
        if cls._model is None:
            if not cls.is_available():
                raise ImportError("sentence-transformers not installed")
            
            # Suppress verbose library logs
            import logging as _logging
            _logging.getLogger("sentence_transformers").setLevel(_logging.WARNING)
            
            from sentence_transformers import SentenceTransformer
            
            logger.debug("[Embeddings] Loading model: %s", cls._model_name)
            cls._model = SentenceTransformer(cls._model_name)
            logger.debug("[Embeddings] Model loaded successfully")
        
        return cls._model
    
    @classmethod
    def embed(cls, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
            Embeddings are L2-normalized for cosine similarity via dot product.
        """
        if not texts:
            return np.array([])
        
        model = cls.get_model()
        return model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    
    @classmethod
    def embed_single(cls, text: str) -> np.ndarray:
        """Embed a single text. Returns 1D array."""
        result = cls.embed([text])
        return result[0] if len(result) > 0 else np.array([])
    
    @classmethod
    def similarity(cls, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Returns:
            Float in [-1, 1], typically [0, 1] for similar content.
        """
        if not text_a or not text_b:
            return 0.0
        
        vecs = cls.embed([text_a, text_b])
        if len(vecs) < 2:
            return 0.0
        
        # Dot product of normalized vectors = cosine similarity
        return float(np.dot(vecs[0], vecs[1]))
    
    @classmethod
    def batch_similarity(cls, query: str, candidates: list[str]) -> list[float]:
        """
        Compute similarity between query and multiple candidates.
        
        More efficient than calling similarity() in a loop.
        
        Returns:
            List of similarity scores, same length as candidates.
        """
        if not query or not candidates:
            return [0.0] * len(candidates)
        
        all_texts = [query] + candidates
        vecs = cls.embed(all_texts)
        
        if len(vecs) < 2:
            return [0.0] * len(candidates)
        
        query_vec = vecs[0]
        candidate_vecs = vecs[1:]
        
        # Batch dot product
        scores = np.dot(candidate_vecs, query_vec)
        return scores.tolist()
    
    @classmethod
    def find_best_match(cls, query: str, candidates: list[str]) -> tuple[int, float]:
        """
        Find the candidate most similar to query.
        
        Returns:
            (best_index, best_score) or (-1, 0.0) if no candidates.
        """
        if not candidates:
            return -1, 0.0
        
        scores = cls.batch_similarity(query, candidates)
        best_idx = int(np.argmax(scores))
        return best_idx, scores[best_idx]


def split_sentences(text: str, max_sentences: int = 50) -> list[str]:
    """
    Simple sentence splitter for quote extraction.
    
    Uses basic punctuation rules. Not perfect but fast.
    """
    import re
    
    if not text:
        return []
    
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter empty and very short
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    return sentences[:max_sentences]


def extract_best_quote(
    claim_text: str,
    content: str,
    *,
    max_len: int = 300,
    min_similarity: float = 0.3,
) -> str | None:
    """
    Extract the most relevant quote from content for a claim.
    
    Args:
        claim_text: The claim to find evidence for
        content: Source content to search
        max_len: Maximum quote length
        min_similarity: Minimum similarity threshold
        
    Returns:
        Best matching sentence/quote, or None if no good match.
    """
    if not EmbedService.is_available():
        return None
    
    sentences = split_sentences(content)
    if not sentences:
        return content[:max_len] if content else None
    
    best_idx, best_score = EmbedService.find_best_match(claim_text, sentences)
    
    if best_idx < 0 or best_score < min_similarity:
        return None
    
    quote = sentences[best_idx]
    return quote[:max_len] if len(quote) > max_len else quote
