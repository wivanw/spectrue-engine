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
Embedding Utilities for ClaimGraph B-Stage

Uses OpenAI text-embedding-3-small for claim similarity.
Includes caching to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING

from spectrue_core.utils.trace import Trace

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS_PER_BATCH = 8000  # Conservative limit for batching
APPROX_CHARS_PER_TOKEN = 4


def _text_hash(text: str) -> str:
    """Generate hash for text caching."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns value in range [-1, 1], where 1 means identical.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class EmbeddingClient:
    """
    Client for generating embeddings with caching.
    
    Uses OpenAI text-embedding-3-small model.
    Caches embeddings by (text_hash, model) to avoid redundant calls.
    Falls back to EmbedService when no client is provided.
    """

    def __init__(self, openai_client: "AsyncOpenAI | None" = None):
        self.client = openai_client
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{EMBEDDING_MODEL}:{_text_hash(text)}"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Uses caching to avoid redundant API calls.
        Batches requests for efficiency.
        Falls back to EmbedService when no async client is available.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []
        empty_texts = 0
        api_error_texts = 0
        fill_none_texts = 0

        # Fallback to EmbedService if no async client
        if self.client is None:
            try:
                from spectrue_core.utils.embedding_service import EmbedService
                if EmbedService.is_available():
                    return EmbedService.embed(texts)
            except ImportError:
                pass
            # Return zero vectors if no service available
            logger.debug("[M72] No embedding client available, returning zero vectors")
            Trace.event(
                "embeddings.zero_vectors",
                {
                    "component": "embedding_client",
                    "model": EMBEDDING_MODEL,
                    "reason": "service_unavailable",
                    "total_texts": len(texts),
                },
            )
            return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

        # Check cache and identify texts that need embedding
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)

        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Empty text gets zero vector
                results[i] = [0.0] * EMBEDDING_DIMENSIONS
                empty_texts += 1
                continue

            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append((i, text.strip()))

        if not texts_to_embed:
            return [r for r in results if r is not None]

        logger.debug("[M72] Embedding %d texts (%d from cache)", 
                     len(texts_to_embed), len(texts) - len(texts_to_embed))

        # Batch texts for API call
        batches = self._create_batches(texts_to_embed)

        for batch in batches:
            batch_indices = [idx for idx, _ in batch]
            batch_texts = [txt for _, txt in batch]

            try:
                response = await self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts,
                )

                # Record embedding cost via meter context
                try:
                    from spectrue_core.billing.meter_context import get_current_llm_meter
                    meter = get_current_llm_meter()
                    if meter:
                        usage = getattr(response, "usage", None)
                        usage_dict = None
                        if usage is not None:
                            usage_dict = {
                                "total_tokens": getattr(usage, "total_tokens", None),
                                "input_tokens": getattr(usage, "prompt_tokens", None),
                            }
                        meter.record_embedding(
                            model=EMBEDDING_MODEL,
                            stage="claim_graph_embed",
                            usage=usage_dict,
                            input_texts=batch_texts,
                            meta={"batch_size": len(batch_texts), "component": "embedding_client"},
                        )
                except Exception:
                    pass  # Metering failure is non-critical

                for i, embedding_data in enumerate(response.data):
                    idx = batch_indices[i]
                    embedding = embedding_data.embedding

                    # Cache the result
                    cache_key = self._cache_key(batch_texts[i])
                    self._cache[cache_key] = embedding

                    results[idx] = embedding

            except Exception as e:
                logger.warning("[M72] Embedding API error: %s", e)
                # Fill with zero vectors on error
                batch_missing = sum(1 for idx in batch_indices if results[idx] is None)
                api_error_texts += batch_missing
                for idx in batch_indices:
                    if results[idx] is None:
                        results[idx] = [0.0] * EMBEDDING_DIMENSIONS

        # Fill any remaining None with zero vectors
        final_results: list[list[float]] = []
        for r in results:
            if r is None:
                fill_none_texts += 1
                final_results.append([0.0] * EMBEDDING_DIMENSIONS)
            else:
                final_results.append(r)

        if empty_texts or api_error_texts or fill_none_texts:
            logger.debug(
                "[M72] Zero vectors summary: total=%d empty=%d api_error=%d fill_none=%d",
                len(texts),
                empty_texts,
                api_error_texts,
                fill_none_texts,
            )
            Trace.event(
                "embeddings.zero_vectors",
                {
                    "component": "embedding_client",
                    "model": EMBEDDING_MODEL,
                    "total_texts": len(texts),
                    "empty_texts": empty_texts,
                    "api_error_texts": api_error_texts,
                    "fill_none_texts": fill_none_texts,
                },
            )

        return final_results

    def _create_batches(
        self, 
        texts_with_indices: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        """
        Create batches of texts for API calls.
        
        Respects token limits for efficient batching.
        """
        batches: list[list[tuple[int, str]]] = []
        current_batch: list[tuple[int, str]] = []
        current_tokens = 0

        for idx, text in texts_with_indices:
            # Estimate tokens (conservative)
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

    def build_similarity_matrix(
        self, 
        embeddings: list[list[float]]
    ) -> list[list[float]]:
        """
        Build NxN similarity matrix from embeddings.
        
        Returns:
            2D matrix where matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
        """
        n = len(embeddings)
        if n == 0:
            return []

        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matrix[i][i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                matrix[i][j] = sim
                matrix[j][i] = sim  # Symmetric

        return matrix

    def get_top_k_similar(
        self,
        target_idx: int,
        similarity_matrix: list[list[float]],
        k: int,
        exclude_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Get top-K most similar indices for a target.
        
        Args:
            target_idx: Index of target claim
            similarity_matrix: Pre-computed similarity matrix
            k: Number of similar items to return
            exclude_indices: Indices to exclude (e.g., self, already selected)
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        if not similarity_matrix or target_idx >= len(similarity_matrix):
            return []

        exclude = exclude_indices or set()
        exclude.add(target_idx)  # Always exclude self

        similarities = [
            (i, sim) 
            for i, sim in enumerate(similarity_matrix[target_idx])
            if i not in exclude
        ]

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
