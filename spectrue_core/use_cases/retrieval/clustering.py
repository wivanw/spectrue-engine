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
Retrieval Clustering Use Case.

Handles clustering of URLs or text content for improved search efficiency.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spectrue_core.adapters.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


@dataclass
class ClusteredItem:
    text: str
    original_index: int
    cluster_id: int | None = None
    similarity_scores: dict[int, float] | None = None


def _stable_cluster_id(urls: list[str]) -> str:
    ordered = sorted([u for u in urls if u])
    raw = "|".join(ordered)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    idx = int(round(q * (len(ordered) - 1)))
    return float(ordered[idx])


def assign_similarity_clusters(
    urls: list[str],
    sim_matrix: list[list[float]],
    *,
    quantile: float,
) -> dict[str, str]:
    """
    Assign URLs to clusters based on similarity matrix and quantile threshold.
    
    Args:
        urls: List of URLs corresponding to rows/cols of sim_matrix.
        sim_matrix: Pre-computed similarity matrix.
        quantile: Quantile to determine similarity threshold.
        
    Returns:
        Dictionary mapping URL to cluster ID.
    """
    if len(urls) <= 1:
        return {urls[0]: _stable_cluster_id(urls)} if urls else {}

    sims: list[float] = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sims.append(float(sim_matrix[i][j]))
    tau = _quantile(sims, quantile)

    adjacency: dict[str, set[str]] = {u: set() for u in urls}
    for i, src in enumerate(urls):
        for j, dst in enumerate(urls):
            if i >= j:
                continue
            if float(sim_matrix[i][j]) >= tau:
                adjacency[src].add(dst)
                adjacency[dst].add(src)

    visited: set[str] = set()
    clusters: dict[str, str] = {}
    for url in urls:
        if url in visited:
            continue
        queue = [url]
        visited.add(url)
        component: list[str] = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        cluster_id = _stable_cluster_id(component)
        for item in component:
            clusters[item] = cluster_id

    return clusters


async def cluster_texts_by_similarity(
    texts: list[str],
    embedding_client: "EmbeddingClient",
    threshold: float = 0.85,
) -> list[list[int]]:
    """
    Cluster texts based on cosine similarity of their embeddings.
    """
    if not texts:
        return []

    embeddings = await embedding_client.embed_texts(texts, purpose="query")
    sim_matrix = embedding_client.build_similarity_matrix(embeddings)
    
    n = len(texts)
    visited = [False] * n
    clusters: list[list[int]] = []
    
    for i in range(n):
        if visited[i]:
            continue
            
        current_cluster = [i]
        visited[i] = True
        
        for j in range(i + 1, n):
            if visited[j]:
                continue
                
            sim = sim_matrix[i][j]
            if sim >= threshold:
                current_cluster.append(j)
                visited[j] = True
        
        clusters.append(current_cluster)
        
    return clusters
