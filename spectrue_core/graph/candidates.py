# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Candidate Generation (B-stage)

Provides similarity-based kNN edges and MST connectivity helpers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from spectrue_core.graph.types import CandidateEdge, ClaimNode

logger = logging.getLogger(__name__)

Edge = Tuple[str, str, float]


def build_knn_edges(
    *,
    nodes: List[ClaimNode],
    similarity_matrix: List[List[float]],
    k: int,
) -> tuple[List[Edge], Dict[str, List[tuple[str, float]]]]:
    """
    Build undirected kNN edges from a similarity matrix.

    Returns:
        (edges, knn_map) where edges are unique (u, v, sim) tuples and knn_map
        lists neighbors per node id.
    """
    edges: List[Edge] = []
    knn_map: Dict[str, List[tuple[str, float]]] = {}
    seen: set[tuple[str, str]] = set()

    for i, node in enumerate(nodes):
        top_similar = sorted(
            [
                (j, similarity_matrix[i][j])
                for j in range(len(nodes))
                if j != i
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        neigh = []
        for j, sim_score in top_similar:
            other = nodes[j]
            pair = tuple(sorted((node.claim_id, other.claim_id)))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append((node.claim_id, other.claim_id, float(sim_score)))
            neigh.append((other.claim_id, float(sim_score)))
        knn_map[node.claim_id] = neigh

    return edges, knn_map


def _prim_mst(nodes: List[str], edges: List[Edge]) -> List[Edge]:
    adj: Dict[str, List[tuple[str, float]]] = {n: [] for n in nodes}
    for u, v, sim in edges:
        dist = 1.0 - float(sim)
        adj[u].append((v, dist))
        adj[v].append((u, dist))

    if not nodes:
        return []

    start = nodes[0]
    in_tree = {start}
    mst: List[Edge] = []

    while len(in_tree) < len(nodes):
        best = None
        for u in list(in_tree):
            for v, dist in adj[u]:
                if v in in_tree:
                    continue
                cand = (dist, u, v)
                if best is None or cand < best:
                    best = cand
        if best is None:
            # Disconnected graph
            break
        dist, u, v = best
        in_tree.add(v)
        sim = 1.0 - dist
        mst.append((u, v, sim))

    return mst


def mst_connectivity(
    *,
    node_ids: List[str],
    candidate_edges: List[Edge],
    similarity_matrix: List[List[float]] | None = None,
    max_nodes_for_full_pairwise: int = 50,
) -> List[Edge]:
    """
    Add MST edges to guarantee connectivity.

    Uses candidate_edges; if they cannot connect all nodes and the graph is
    small, falls back to full pairwise distances for MST only.
    """
    mst = _prim_mst(node_ids, candidate_edges)
    if len(mst) >= max(0, len(node_ids) - 1):
        return mst

    if similarity_matrix is None or len(node_ids) > max_nodes_for_full_pairwise:
        return mst

    full_edges: List[Edge] = []
    for i, src in enumerate(node_ids):
        for j in range(i + 1, len(node_ids)):
            sim = similarity_matrix[i][j]
            full_edges.append((src, node_ids[j], float(sim)))

    return _prim_mst(node_ids, full_edges)


async def generate_candidate_edges(
    *,
    nodes: list[ClaimNode],
    config,
    embedding_client,
) -> list[CandidateEdge]:
    """
    Legacy-compatible candidate generator that now builds kNN edges only.

    Returns CandidateEdge objects for compatibility; adjacency/topic caps
    are removed in favor of deterministic similarity kNN.
    """
    if len(nodes) < 2:
        return []

    texts = [n.text for n in nodes]
    embeddings = await embedding_client.embed_texts(texts)
    sim_matrix = embedding_client.build_similarity_matrix(embeddings)

    knn_edges, _ = build_knn_edges(nodes=nodes, similarity_matrix=sim_matrix, k=getattr(config, "k_sim", len(nodes)))
    candidates: list[CandidateEdge] = []
    for src, dst, sim_score in knn_edges:
        same_section = False
        try:
            src_node = next(n for n in nodes if n.claim_id == src)
            dst_node = next(n for n in nodes if n.claim_id == dst)
            same_section = src_node.section_id == dst_node.section_id
        except StopIteration:
            same_section = False
        candidates.append(
            CandidateEdge(
                src_id=src,
                dst_id=dst,
                reason="sim",
                sim_score=sim_score,
                same_section=same_section,
                cross_topic=False,
            )
        )

    logger.debug(
        "[M109] B-Stage: %d similarity candidate edges from %d nodes",
        len(candidates),
        len(nodes),
    )
    return candidates
