# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Embedding Service (OpenAI)

Provides semantic similarity using OpenAI embeddings.
Model: text-embedding-3-small (1536 dims)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS_PER_BATCH = 8000  # Conservative limit for batching
APPROX_CHARS_PER_TOKEN = 4


def _text_hash(text: str) -> str:
    """Generate hash for text caching."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector for cosine similarity via dot product."""
    if not vec:
        return []
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class EmbedService:
    """
    Singleton embedding service using OpenAI embeddings.

    Lazy initializes the OpenAI client and caches embeddings by (text_hash, model).
    """

    _client: "OpenAI | None" = None
    _api_key: str | None = None
    _available: bool | None = None
    _cache: dict[str, list[float]] = {}

    @classmethod
    def configure(
        cls,
        *,
        openai_api_key: str | None = None,
        client: "OpenAI | None" = None,
    ) -> None:
        """Configure the OpenAI client and API key (optional)."""
        if openai_api_key is not None:
            if isinstance(openai_api_key, str) and openai_api_key.strip():
                cls._api_key = openai_api_key.strip()
            else:
                cls._api_key = None
            cls._client = None
        if client is not None:
            cls._client = client
        cls._available = None

    @classmethod
    def _resolve_api_key(cls) -> str | None:
        return (
            cls._api_key
            or os.getenv("SPECTRUE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI embeddings are available (client or API key)."""
        if cls._client is not None:
            return True
        if cls._available is None:
            if (os.getenv("SPECTRUE_TEST_OFFLINE") or "").strip().lower() in {
                "1", "true", "yes", "y", "on"
            }:
                cls._available = False
                logger.debug("[Embeddings] Offline test mode enabled")
                return cls._available
            try:
                import openai  # noqa: F401
            except ImportError:
                cls._available = False
                logger.debug("[Embeddings] openai not installed")
                return cls._available
            api_key = cls._resolve_api_key()
            cls._available = bool(api_key)
            if not api_key:
                logger.debug("[Embeddings] OpenAI API key not configured")
        return cls._available

    @classmethod
    def _get_client(cls) -> "OpenAI":
        if cls._client is None:
            if not cls.is_available():
                raise RuntimeError("OpenAI embeddings not available")
            from openai import OpenAI
            api_key = cls._resolve_api_key()
            if api_key:
                cls._client = OpenAI(api_key=api_key)
            else:
                cls._client = OpenAI()
        return cls._client

    @classmethod
    def _cache_key(cls, text: str) -> str:
        return f"{EMBEDDING_MODEL}:{_text_hash(text)}"

    @classmethod
    def _create_batches(
        cls,
        texts_with_indices: list[tuple[int, str]],
    ) -> list[list[tuple[int, str]]]:
        batches: list[list[tuple[int, str]]] = []
        current_batch: list[tuple[int, str]] = []
        current_tokens = 0

        for idx, text in texts_with_indices:
            text_tokens = len(text) // APPROX_CHARS_PER_TOKEN + 1
            if current_tokens + text_tokens > MAX_TOKENS_PER_BATCH and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append((idx, text))
            current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    @classmethod
    def embed(cls, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts via OpenAI.

        Returns:
            List of embedding vectors (same order as input).
            Embeddings are L2-normalized for cosine similarity via dot product.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if not text or not str(text).strip():
                results[i] = [0.0] * EMBEDDING_DIMENSIONS
                continue

            cleaned = str(text).strip()
            cache_key = cls._cache_key(cleaned)
            if cache_key in cls._cache:
                results[i] = cls._cache[cache_key]
            else:
                texts_to_embed.append((i, cleaned))

        if texts_to_embed:
            client = cls._get_client()
            batches = cls._create_batches(texts_to_embed)

            for batch in batches:
                batch_indices = [idx for idx, _ in batch]
                batch_texts = [txt for _, txt in batch]

                try:
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts,
                    )
                    for i, embedding_data in enumerate(response.data):
                        idx = batch_indices[i]
                        embedding = _normalize(list(embedding_data.embedding))
                        cache_key = cls._cache_key(batch_texts[i])
                        cls._cache[cache_key] = embedding
                        results[idx] = embedding
                except Exception as e:
                    logger.warning("[Embeddings] Embedding API error: %s", e)
                    for idx in batch_indices:
                        if results[idx] is None:
                            results[idx] = [0.0] * EMBEDDING_DIMENSIONS

        return [
            r if r is not None else [0.0] * EMBEDDING_DIMENSIONS
            for r in results
        ]

    @classmethod
    def embed_single(cls, text: str) -> list[float]:
        """Embed a single text. Returns a 1D vector."""
        result = cls.embed([text])
        return result[0] if len(result) > 0 else []

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

        return float(_dot(vecs[0], vecs[1]))

    @classmethod
    def batch_similarity(cls, query: str, candidates: list[str]) -> list[float]:
        """
        Compute similarity between query and multiple candidates.

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

        scores = [_dot(vec, query_vec) for vec in candidate_vecs]
        return [float(score) for score in scores]

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
        best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
        return int(best_idx), float(scores[best_idx])

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the embedding cache."""
        cls._cache.clear()


def split_sentences(text: str, max_sentences: int = 50) -> list[str]:
    """
    Simple sentence splitter for quote extraction.

    Uses basic punctuation rules. Not perfect but fast.
    """
    import re

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
