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
    from spectrue_core.billing.metering import LLMMeter

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


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    """Compute mean pooling of a list of vectors."""
    if not vectors:
        return []
    if not vectors[0]:
        return []
    
    dim = len(vectors[0])
    avg = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            continue
        for i, val in enumerate(vec):
            avg[i] += val
    
    count = len(vectors)
    if count == 0:
        return [0.0] * dim
        
    return [x / count for x in avg]


def _chunk_text(text: str, chunk_size: int = 8000) -> list[str]:
    """Split text into chunks of max chunk_size characters."""
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


class EmbedService:
    """
    Singleton embedding service using OpenAI embeddings.

    Lazy initializes the OpenAI client and caches embeddings by (text_hash, model).
    """

    _client: "OpenAI | None" = None
    _api_key: str | None = None
    _available: bool | None = None
    _cache: dict[str, list[float]] = {}
    _meter: "LLMMeter | None" = None
    _meter_stage: str = "embed"

    @classmethod
    def configure(
        cls,
        *,
        openai_api_key: str | None = None,
        client: "OpenAI | None" = None,
        meter: "LLMMeter | None" = None,
        meter_stage: str | None = None,
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
        cls._meter = meter
        if meter_stage is not None:
            cls._meter_stage = str(meter_stage)

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
    def embed(
        cls, 
        texts: list[str], 
        *, 
        purpose: str = "quote"  # "quote" | "document" | "query"
    ) -> list[list[float]]:
        """
        Embed a list of texts via OpenAI.
        
        Args:
            texts: List of strings to embed.
            purpose: Semantic intent:
                - "quote"/"query": Short text (sentence/paragraph). Limit: 2000 chars.
                - "document"/"indexing": Large text. Limit: 8000 chars (safe chunking).

        Returns:
            List of embedding vectors (same order as input).
            Embeddings are L2-normalized for cosine similarity via dot product.
        """
        import inspect

        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []
        empty_texts = 0
        api_error_texts = 0
        zero_norm_texts = 0
        fill_none_texts = 0

        # Define limits based on purpose
        # "quote" should be atomic. "document" can be chunky.
        match purpose:
            case "quote" | "query":
                # Strict limit for atomic units. 
                # If exceeded, it means upstream splitting failed.
                soft_limit = 2000
            case "document":
                # Larger limit for docs, but still constrained by model window (8192 tokens)
                soft_limit = 8000
            case "indexing":
                # Massive limit for full corpus indexing. Relies on internal chunking/pooling.
                soft_limit = 2_000_000
            case _:
                # Default strict
                soft_limit = 1000

        for i, text in enumerate(texts):
            if not text or not str(text).strip():
                results[i] = [0.0] * EMBEDDING_DIMENSIONS
                empty_texts += 1
                continue

            cleaned = str(text).strip()
            
            # Contract Enforcement: Check soft limit violation
            if len(cleaned) > soft_limit:
                 # Diagnose caller
                try:
                    stack = inspect.stack()
                    # frame 0 is this, 1 is caller
                    caller_frame = stack[1]
                    caller_name = f"{caller_frame.filename.split('/')[-1]}:{caller_frame.function}:{caller_frame.lineno}"
                except Exception:
                    caller_name = "unknown"

                logger.warning(
                    "[Embeddings] ⚠️ Contract Violation: purpose='%s' text length=%d > %d. Caller: %s. Proceeding with chunking fallback.",
                    purpose, len(cleaned), soft_limit, caller_name
                )
            
            # Handle large texts via Chunking + Average Pooling
            # Limit is ~8192 tokens. For dense text/Cyrillic, 1 char ~ 1 token.
            # Using 8000 chars as HARD SAFETY limit for API 400 prevention.
            HARD_LIMIT = 8000
            
            if len(cleaned) > HARD_LIMIT:
                cache_key = cls._cache_key(cleaned)
                if cache_key in cls._cache:
                    results[i] = cls._cache[cache_key]
                else:
                    # Recursive chunk processing
                    chunks = _chunk_text(cleaned, HARD_LIMIT)
                    logger.debug(
                        "[Embeddings] Chunking large text length=%d into %d chunks", 
                        len(cleaned), len(chunks)
                    )
                    
                    # Embed chunks (recursive with same purpose to allow drill-down or default)
                    # Use 'document' purpose for chunks to suppress warnings for the recursive call if they are large? 
                    # Actually chunks are guaranteed < 8000. So OK.
                    chunk_vecs = cls.embed(chunks, purpose="document")
                    
                    # Mean pool and normalize
                    avg_vec = _mean_pool(chunk_vecs)
                    final_vec = _normalize(avg_vec)
                    
                    # Cache the aggregated vector
                    cls._cache[cache_key] = final_vec
                    results[i] = final_vec
                continue

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
                    # Meter embeddings cost if available.
                    try:
                        usage = getattr(response, "usage", None)
                        usage_dict = None
                        if isinstance(usage, dict):
                            usage_dict = usage
                        elif usage is not None:
                            usage_dict = {
                                "total_tokens": getattr(usage, "total_tokens", None),
                                "input_tokens": getattr(usage, "prompt_tokens", None),
                            }
                        if cls._meter is not None:
                            cls._meter.record_embedding(
                                model=EMBEDDING_MODEL,
                                stage=cls._meter_stage,
                                usage=usage_dict,
                                input_texts=batch_texts,
                                meta={
                                    "batch_size": len(batch_texts),
                                    "cached": False,
                                },
                            )
                    except Exception as me:
                        logger.debug("[Embeddings] metering skipped: %s", me)

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
    def embed_single(cls, text: str, *, purpose: str = "quote") -> list[float]:
        """Embed a single text. Returns a 1D vector."""
        result = cls.embed([text], purpose=purpose)
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

        # Usually comparing short texts (claims, sentences)
        vecs = cls.embed([text_a, text_b], purpose="quote")
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
        # Query and candidates are usually short
        vecs = cls.embed(all_texts, purpose="query")

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
    Robust sentence splitter for quote extraction.
    
    Strategies:
    1. Split by newlines (paragraphs).
    2. Split by punctuation (.!?).
    3. Hard chop if segment > 2000 chars (prevents huge blobs).
    """
    import re
    
    if not text:
        return []

    # 1. Split by newlines first (logical paragraphs)
    # This handles cases where text is a list of headers without punctuation.
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    final_sentences = []
    final_sentences = []
    MAX_CHARS_PER_SEGMENT = 1000
    
    for p in paragraphs:
        if len(final_sentences) >= max_sentences:
            break
            
        # 2. Split by punctuation
        # Using a safer regex that doesn't consume the delimiter
        # But keeping consistent with previous logic: split ON whitespace AFTER punctuation
        sents = re.split(r"(?<=[.!?])\s+", p)
        
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) < 20:  # Skip tiny fragments
                continue
            
            # 3. Hard limit check
            if len(s_clean) > MAX_CHARS_PER_SEGMENT:
                # If a "sentence" is huge, it's likely not a sentence but unparsed blob.
                # Chunk it blindly into quote-sized pieces.
                # Reuse _chunk_text helper if available, or slice manually
                chunks = [s_clean[i:i+MAX_CHARS_PER_SEGMENT] for i in range(0, len(s_clean), MAX_CHARS_PER_SEGMENT)]
                final_sentences.extend(chunks)
            else:
                final_sentences.append(s_clean)
                
            if len(final_sentences) >= max_sentences:
                break
                
    return final_sentences[:max_sentences]


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
    embeddings = EmbedService.embed(flat_texts, purpose="quote")

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

