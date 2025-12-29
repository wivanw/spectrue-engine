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

from spectrue_core.utils.trace import Trace

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
        empty_texts = 0
        api_error_texts = 0
        zero_norm_texts = 0
        fill_none_texts = 0

        for i, text in enumerate(texts):
            if not text or not str(text).strip():
                results[i] = [0.0] * EMBEDDING_DIMENSIONS
                empty_texts += 1
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
                        if not embedding or not any(embedding):
                            zero_norm_texts += 1
                        cache_key = cls._cache_key(batch_texts[i])
                        cls._cache[cache_key] = embedding
                        results[idx] = embedding
                except Exception as e:
                    logger.warning("[Embeddings] Embedding API error: %s", e)
                    batch_missing = sum(1 for idx in batch_indices if results[idx] is None)
                    api_error_texts += batch_missing
                    for idx in batch_indices:
                        if results[idx] is None:
                            results[idx] = [0.0] * EMBEDDING_DIMENSIONS

        final_results: list[list[float]] = []
        for r in results:
            if r is None:
                fill_none_texts += 1
                final_results.append([0.0] * EMBEDDING_DIMENSIONS)
            else:
                final_results.append(r)

        if empty_texts or api_error_texts or zero_norm_texts or fill_none_texts:
            logger.debug(
                "[Embeddings] Zero vectors summary: total=%d empty=%d api_error=%d zero_norm=%d fill_none=%d",
                len(texts),
                empty_texts,
                api_error_texts,
                zero_norm_texts,
                fill_none_texts,
            )
            Trace.event(
                "embeddings.zero_vectors",
                {
                    "component": "embed_service",
                    "model": EMBEDDING_MODEL,
                    "total_texts": len(texts),
                    "empty_texts": empty_texts,
                    "api_error_texts": api_error_texts,
                    "zero_norm_texts": zero_norm_texts,
                    "fill_none_texts": fill_none_texts,
                },
            )

        return final_results

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


async def extract_best_quote_async(
    claim_text: str,
    content: str,
    *,
    max_len: int = 300,
    min_similarity: float = 0.3,
) -> str | None:
    """
    Async version of extract_best_quote.
    
    Runs embedding operations in thread pool to avoid blocking event loop.
    """
    import asyncio
    import concurrent.futures
    from functools import partial
    
    if not EmbedService.is_available():
        return None
    
    loop = asyncio.get_running_loop()
    
    # Use shared thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor,
            partial(extract_best_quote, claim_text, content, max_len=max_len, min_similarity=min_similarity),
        )
    
    return result


def extract_best_quotes_batch(
    items: list[tuple[str, str]],
    *,
    max_len: int = 300,
    min_similarity: float = 0.3,
) -> list[str | None]:
    """
    Batch extract quotes for multiple (claim_text, content) pairs.
    
    More efficient than calling extract_best_quote() N times because
    it batches all embeddings into a single API call.
    
    Args:
        items: List of (claim_text, content) tuples
        max_len: Maximum quote length
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of best quotes (or None) for each item
    """
    if not items:
        return []
    
    if not EmbedService.is_available():
        Trace.event("embeddings.caller", {
            "caller": "extract_best_quotes_batch",
            "action": "skip",
            "reason": "embed_service_unavailable",
        })
        return [None] * len(items)
    
    Trace.event("embeddings.caller", {
        "caller": "extract_best_quotes_batch",
        "action": "start",
        "items_count": len(items),
    })
    
    # Pre-process: split sentences for each content
    all_sentences: list[list[str]] = []
    claim_texts: list[str] = []
    
    for claim_text, content in items:
        sentences = split_sentences(content) if content else []
        all_sentences.append(sentences)
        claim_texts.append(claim_text)
    
    # Flatten all texts for single batch embedding
    # Structure: [claim1, sent1_1, sent1_2, ..., claim2, sent2_1, ...]
    flat_texts: list[str] = []
    offsets: list[tuple[int, int, int]] = []  # (claim_idx, sentence_start, sentence_count)
    
    for i, (claim_text, sentences) in enumerate(zip(claim_texts, all_sentences)):
        claim_idx = len(flat_texts)
        flat_texts.append(claim_text)
        sent_start = len(flat_texts)
        flat_texts.extend(sentences)
        offsets.append((claim_idx, sent_start, len(sentences)))
    
    if not flat_texts:
        return [None] * len(items)
    
    # Single batch embedding call
    embeddings = EmbedService.embed(flat_texts)
    
    if not embeddings:
        return [None] * len(items)
    
    # Compute similarities and find best quotes
    results: list[str | None] = []
    
    for i, (claim_idx, sent_start, sent_count) in enumerate(offsets):
        if sent_count == 0:
            # No sentences - return truncated content or None
            content = items[i][1]
            results.append(content[:max_len] if content else None)
            continue
        
        claim_vec = embeddings[claim_idx]
        best_score = -1.0
        best_sent_idx = -1
        
        for j in range(sent_count):
            sent_vec = embeddings[sent_start + j]
            score = float(_dot(claim_vec, sent_vec))
            if score > best_score:
                best_score = score
                best_sent_idx = j
        
        if best_sent_idx < 0 or best_score < min_similarity:
            results.append(None)
        else:
            quote = all_sentences[i][best_sent_idx]
            results.append(quote[:max_len] if len(quote) > max_len else quote)
    
    non_null = sum(1 for r in results if r is not None)
    Trace.event("embeddings.caller.result", {
        "caller": "extract_best_quotes_batch",
        "total": len(results),
        "non_null": non_null,
    })
    
    return results


async def extract_best_quotes_batch_async(
    items: list[tuple[str, str]],
    *,
    max_len: int = 300,
    min_similarity: float = 0.3,
) -> list[str | None]:
    """
    Async batch quote extraction.
    
    Runs in thread pool to avoid blocking event loop.
    """
    import asyncio
    import concurrent.futures
    from functools import partial
    
    if not items:
        return []
    
    if not EmbedService.is_available():
        return [None] * len(items)
    
    loop = asyncio.get_running_loop()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor,
            partial(extract_best_quotes_batch, items, max_len=max_len, min_similarity=min_similarity),
        )
    
    return result

