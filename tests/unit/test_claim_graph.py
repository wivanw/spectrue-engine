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
ClaimGraph orchestration tests (M109).

Focus on:
- MST connectivity guarantee
- Personalized PageRank honoring teleport priors
- Budgeted submodular selection without hard caps
- Dominant claim inclusion via selection
- Quality gates degrade confidence without disabling graph
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from spectrue_core.graph.candidates import mst_connectivity
from spectrue_core.graph.claim_graph import ClaimGraphBuilder
from spectrue_core.graph.ranking import compute_pagerank_with_ranks
from spectrue_core.graph.selection import greedy_budgeted_submodular
from spectrue_core.runtime_config import ClaimGraphConfig


class DummyEmbeddingClient:
    """Stub embedding client returning a fixed similarity matrix."""

    def __init__(self, sim_matrix):
        self.sim_matrix = sim_matrix

    async def embed_texts(self, texts, **kwargs):
        return [[0.0] * 3 for _ in texts]

    def build_similarity_matrix(self, embeddings):
        assert len(embeddings) == len(self.sim_matrix)
        return self.sim_matrix


@dataclass
class DummyOpenAI:
    """Placeholder to satisfy builder signature."""
    pass


def make_config(**overrides) -> ClaimGraphConfig:
    cfg = ClaimGraphConfig()
    return replace(cfg, **overrides)


def test_mst_connectivity_bridge():
    """MST adds edges to connect components when kNN is sparse."""
    nodes = ["c1", "c2", "c3"]
    candidate_edges = [("c1", "c2", 0.9)]  # c3 disconnected without MST
    sim_matrix = [
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ]

    mst = mst_connectivity(
        node_ids=nodes,
        candidate_edges=candidate_edges,
        similarity_matrix=sim_matrix,
        max_nodes_for_full_pairwise=50,
    )

    assert len(mst) >= len(nodes) - 1
    assert any("c3" in {u, v} for u, v, _ in mst)


def test_personalized_pagerank_respects_prior():
    """Large teleport prior should push node to top rank."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b", 1.0), ("b", "c", 1.0)]
    teleport = {"a": 0.1, "b": 5.0, "c": 0.1}

    pr, ranks = compute_pagerank_with_ranks(
        node_ids=nodes,
        edges=edges,
        teleport=teleport,
        alpha=0.85,
        eps=1e-9,
        max_iter=100,
    )

    assert pr["b"] > pr["a"]
    assert pr["b"] > pr["c"]
    assert ranks["b"] == 0


def test_budgeted_selection_scales_with_budget():
    """Higher budget selects more items; no hidden hard cap."""
    nodes = ["c1", "c2", "c3", "c4"]

    def sim(a: str, b: str) -> float:
        return 1.0 if a == b else 0.2

    pagerank = {n: 0.25 for n in nodes}
    cost = {n: 1.0 for n in nodes}

    low_selected, _ = greedy_budgeted_submodular(
        nodes=nodes,
        sim=sim,
        pagerank=pagerank,
        cost=cost,
        budget=1.0,
        lambda_rank=0.3,
        mu_redundancy=0.1,
    )
    high_selected, _ = greedy_budgeted_submodular(
        nodes=nodes,
        sim=sim,
        pagerank=pagerank,
        cost=cost,
        budget=3.0,
        lambda_rank=0.3,
        mu_redundancy=0.1,
    )

    assert len(high_selected) > len(low_selected)
    assert len(high_selected) <= len(nodes)


@pytest.mark.asyncio
async def test_selection_includes_dominant_claim(tmp_path):
    """Dominant claim from fixture should be selected and ranked near top."""
    fixture = Path(__file__).parents[1] / "fixtures" / "graph" / "kerosene_dominant_case.json"
    data = json.loads(fixture.read_text())
    claims = data["claims"]

    sim_matrix = [
        [1.0, 0.2, 0.1, 0.2],  # c1
        [0.2, 1.0, 0.1, 0.9],  # c2 (dominant)
        [0.1, 0.1, 1.0, 0.05],  # c3
        [0.2, 0.9, 0.05, 1.0],  # c4
    ]

    builder = ClaimGraphBuilder(
        config=make_config(k_sim=2, selection_budget=3.0),
        openai_client=DummyOpenAI(),
        embedding_client=DummyEmbeddingClient(sim_matrix),
    )

    result = await builder.build(claims)
    selected_ids = {step["id"] for step in result.selection_trace}
    top_ranked = [r.claim_id for r in result.all_ranked[:3]]

    assert data["expected"]["dominant_claim_id"] in selected_ids
    assert data["expected"]["dominant_claim_id"] in top_ranked


@pytest.mark.asyncio
async def test_quality_gate_degrades_not_kills():
    """Low-density graph scales confidence but still returns selections."""
    claims = [
        {"id": "c1", "text": "Alpha"},
        {"id": "c2", "text": "Beta"},
    ]
    sim_matrix = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    builder = ClaimGraphBuilder(
        config=make_config(k_sim=1, min_kept_ratio=0.2, selection_budget=1.0),
        openai_client=DummyOpenAI(),
        embedding_client=DummyEmbeddingClient(sim_matrix),
    )

    result = await builder.build(claims)

    assert result.confidence_scalar < 1.0
    assert result.disabled is False
    assert len(result.selection_trace) > 0
