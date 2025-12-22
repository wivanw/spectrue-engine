# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Candidate Generation (B-stage)

Isolated from `ClaimGraphBuilder` so the builder reads as a pipeline of steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spectrue_core.graph.types import CandidateEdge, ClaimNode

if TYPE_CHECKING:
    from spectrue_core.graph.embedding_util import EmbeddingClient
    from spectrue_core.runtime_config import ClaimGraphConfig

logger = logging.getLogger(__name__)


async def generate_candidate_edges(
    *,
    nodes: list[ClaimNode],
    config: "ClaimGraphConfig",
    embedding_client: "EmbeddingClient",
) -> list[CandidateEdge]:
    """
    Generate candidate edges using embedding similarity and adjacency.

    MVP: sim + adjacent (keyword overlap is stub/disabled)
    """
    if len(nodes) < 2:
        return []

    texts = [n.text for n in nodes]
    embeddings = await embedding_client.embed_texts(texts)
    sim_matrix = embedding_client.build_similarity_matrix(embeddings)

    candidates: list[CandidateEdge] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, node in enumerate(nodes):
        node_candidates: list[CandidateEdge] = []

        # Similarity-based candidates (Top-K_SIM)
        # M74: Topic-aware check
        top_similar = embedding_client.get_top_k_similar(i, sim_matrix, k=len(nodes))

        sim_count = 0
        for j, sim_score in top_similar:
            if sim_count >= config.k_sim:
                break

            other = nodes[j]

            # M74: Skip cross-topic for SIM edges
            if config.topic_aware and node.topic_key != other.topic_key:
                continue

            pair = tuple(sorted([node.claim_id, other.claim_id]))
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            node_candidates.append(
                CandidateEdge(
                    src_id=node.claim_id,
                    dst_id=other.claim_id,
                    reason="sim",
                    sim_score=sim_score,
                    same_section=node.section_id == other.section_id,
                    cross_topic=False,  # SIM edges are always within-topic here
                )
            )
            sim_count += 1

        # Adjacency-based candidates (±K_ADJ in same section)
        for delta in range(-config.k_adj, config.k_adj + 1):
            if delta == 0:
                continue

            adj_idx = i + delta
            if adj_idx < 0 or adj_idx >= len(nodes):
                continue

            other = nodes[adj_idx]
            if node.section_id != other.section_id:
                continue

            pair = tuple(sorted([node.claim_id, other.claim_id]))
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            is_cross_topic = node.topic_key != other.topic_key
            node_candidates.append(
                CandidateEdge(
                    src_id=node.claim_id,
                    dst_id=other.claim_id,
                    reason="adjacent",
                    sim_score=0.0,
                    same_section=True,
                    cross_topic=is_cross_topic,
                )
            )

        # Apply cap per node
        if len(node_candidates) > config.k_total_cap:
            node_candidates.sort(key=lambda e: e.sim_score, reverse=True)
            node_candidates = node_candidates[: config.k_total_cap]

        candidates.extend(node_candidates)

    logger.debug(
        "[M72] B-Stage: %d candidate edges from %d nodes", len(candidates), len(nodes)
    )
    return candidates

