# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Ranking (personalized PageRank).

Implements deterministic personalized PageRank with teleport weights derived
from node priors and explicit handling of dangling nodes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def personalized_pagerank(
    nodes: List[str],
    out_edges: Dict[str, List[Tuple[str, float]]],
    teleport: Dict[str, float],
    alpha: float = 0.85,
    eps: float = 1e-8,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Deterministic personalized PageRank."""
    n = len(nodes)
    if n == 0:
        return {}
    pr = {u: 1.0 / n for u in nodes}

    trans: Dict[str, List[Tuple[str, float]]] = {}
    for u in nodes:
        lst = out_edges.get(u, [])
        s = sum(max(0.0, w) for _, w in lst)
        if s <= 0:
            trans[u] = []
        else:
            trans[u] = [(v, max(0.0, w) / s) for v, w in lst]

    # Normalize teleport vector; if empty use uniform.
    tel_sum = sum(max(0.0, v) for v in teleport.values())
    if tel_sum <= 0:
        teleport = {u: 1.0 / n for u in nodes}
    else:
        teleport = {u: max(0.0, teleport.get(u, 0.0)) / tel_sum for u in nodes}

    for _ in range(max_iter):
        new_pr = {u: (1.0 - alpha) * teleport.get(u, 0.0) for u in nodes}
        for u in nodes:
            pu = pr[u]
            outs = trans[u]
            if not outs:
                # Dangling node: redistribute along teleport vector
                for v in nodes:
                    new_pr[v] += alpha * pu * teleport.get(v, 0.0)
                continue
            for v, p_uv in outs:
                new_pr[v] += alpha * pu * p_uv

        delta = sum(abs(new_pr[u] - pr[u]) for u in nodes)
        pr = new_pr
        if delta < eps:
            break

    s = sum(pr.values())
    if s > 0:
        pr = {u: pr[u] / s for u in nodes}
    return pr


def compute_pagerank_with_ranks(
    *,
    node_ids: List[str],
    edges: List[Tuple[str, str, float]],
    teleport: Dict[str, float],
    alpha: float,
    eps: float,
    max_iter: int,
    undirected: bool = True,
) -> tuple[Dict[str, float], Dict[str, int]]:
    """Run personalized PageRank and return scores + centrality ranks."""
    out_edges: Dict[str, List[Tuple[str, float]]] = {n: [] for n in node_ids}
    for src, dst, weight in edges:
        w = max(0.0, float(weight))
        out_edges.setdefault(src, []).append((dst, w))
        if undirected:
            out_edges.setdefault(dst, []).append((src, w))

    pr = personalized_pagerank(
        node_ids, out_edges, teleport, alpha=alpha, eps=eps, max_iter=max_iter
    )
    ranked_ids = sorted(node_ids, key=lambda n: pr.get(n, 0.0), reverse=True)
    centrality_rank = {cid: idx for idx, cid in enumerate(ranked_ids)}
    return pr, centrality_rank
