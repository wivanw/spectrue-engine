# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Ranking

Isolated ranking logic from `ClaimGraphBuilder` so the builder reads as a
sequence of steps.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from spectrue_core.graph.types import (
    EdgeRelation,
    RELATION_MULTIPLIERS,
    STRUCTURAL_RELATIONS,
    ClaimNode,
    RankedClaim,
    TypedEdge,
)

if TYPE_CHECKING:
    from spectrue_core.runtime_config import ClaimGraphConfig

logger = logging.getLogger(__name__)


def rank_claims_pagerank(
    *,
    nodes: list[ClaimNode],
    edges: list[TypedEdge],
    config: "ClaimGraphConfig",
) -> list[RankedClaim]:
    """
    Rank claims using PageRank and structural weight.

    Uses networkx for PageRank computation (optional dependency).
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("[M72] networkx not installed, skipping ranking")
        return [
            RankedClaim(
                claim_id=n.claim_id,
                centrality_score=n.importance,
                in_structural_weight=0.0,
                in_contradict_weight=0.0,
                is_key_claim=i < config.top_k,
            )
            for i, n in enumerate(sorted(nodes, key=lambda x: x.importance, reverse=True))
        ]

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.claim_id, importance=node.importance)

    for edge in edges:
        weight = edge.score * RELATION_MULTIPLIERS.get(edge.relation, 0.0)
        if weight > 0:
            G.add_edge(edge.src_id, edge.dst_id, weight=weight, relation=edge.relation)

    try:
        pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
    except Exception as e:
        logger.warning("[M72] PageRank failed: %s", e)
        pagerank = {n.claim_id: n.importance for n in nodes}

    structural_weights: dict[str, float] = defaultdict(float)
    contradict_weights: dict[str, float] = defaultdict(float)
    for edge in edges:
        weight = edge.score * RELATION_MULTIPLIERS.get(edge.relation, 0.0)
        if edge.relation in STRUCTURAL_RELATIONS:
            structural_weights[edge.dst_id] += weight
        elif edge.relation == EdgeRelation.CONTRADICTS:
            contradict_weights[edge.dst_id] += weight

    ranked: list[RankedClaim] = [
        RankedClaim(
            claim_id=node.claim_id,
            centrality_score=pagerank.get(node.claim_id, 0.0),
            in_structural_weight=structural_weights.get(node.claim_id, 0.0),
            in_contradict_weight=contradict_weights.get(node.claim_id, 0.0),
            is_key_claim=False,
        )
        for node in nodes
    ]

    max_centrality = max((r.centrality_score for r in ranked), default=1.0) or 1.0
    max_structural = max((r.in_structural_weight for r in ranked), default=1.0) or 1.0

    combined_scores: dict[str, float] = {}
    for r in ranked:
        combined_scores[r.claim_id] = (
            r.centrality_score / max_centrality * 0.6
            + r.in_structural_weight / max_structural * 0.4
        )

    ranked.sort(key=lambda r: combined_scores.get(r.claim_id, 0.0), reverse=True)
    for i, r in enumerate(ranked):
        r.is_key_claim = i < config.top_k

    logger.debug(
        "[M72] Ranking: %d claims, top %d selected as key",
        len(ranked),
        min(len(ranked), config.top_k),
    )
    return ranked

