# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Budgeted submodular selection for claim graphs.

Implements a facility-location style objective with redundancy penalty:
coverage gain from similarity + rank bonus from PageRank − redundancy penalty,
optimized greedily under a budget.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple


def greedy_budgeted_submodular(
    *,
    nodes: List[str],
    sim: Callable[[str, str], float],
    pagerank: Dict[str, float],
    cost: Dict[str, float],
    budget: float,
    lambda_rank: float = 0.3,
    mu_redundancy: float = 0.1,
) -> tuple[List[str], List[dict]]:
    """
    Greedy selection that maximizes (coverage + lambda_rank * PageRank - redundancy) / cost.

    Args:
        nodes: Ordered node identifiers to consider.
        sim: Similarity function returning a deterministic float in [0,1].
        pagerank: Node -> pagerank weight.
        cost: Node -> cost (non-negative).
        budget: Total allowable cost (non-negative).
        lambda_rank: Weight for pagerank bonus.
        mu_redundancy: Weight for redundancy penalty.

    Returns:
        (selected_ids, trace_steps) where trace_steps contains per-step gain/cost/remaining.
    """
    selected: List[str] = []
    selected_set: Set[str] = set()
    remaining = max(0.0, float(budget))
    best_cov: Dict[str, float] = {i: 0.0 for i in nodes}
    trace: List[dict] = []

    def delta_gain(x: str) -> float:
        cov_delta = 0.0
        for i in nodes:
            s_ix = max(0.0, sim(i, x))
            if s_ix > best_cov[i]:
                cov_delta += s_ix - best_cov[i]
        rank_bonus = lambda_rank * pagerank.get(x, 0.0)
        redundancy = 0.0
        for y in selected:
            redundancy += max(0.0, sim(x, y))
        return cov_delta + rank_bonus - mu_redundancy * redundancy

    while True:
        best = None
        for x in nodes:
            if x in selected_set:
                continue
            c = float(cost.get(x, 0.0))
            if c <= 0:
                c = 1e-9
            if c > remaining:
                continue
            g = delta_gain(x)
            if g <= 0:
                continue
            score = g / c
            cand = (score, g, -c, x)
            if best is None or cand > best:
                best = cand

        if best is None:
            break

        _, g, neg_c, x = best
        c = -neg_c
        selected.append(x)
        selected_set.add(x)
        remaining -= c
        for i in nodes:
            s_ix = max(0.0, sim(i, x))
            if s_ix > best_cov[i]:
                best_cov[i] = s_ix
        trace.append(
            {
                "id": x,
                "gain": g,
                "cost": c,
                "remaining_budget": max(0.0, remaining),
            }
        )

    return selected, trace

