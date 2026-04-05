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
ClaimGraph orchestration without heuristics.

Deterministic flow:
- compute embeddings and pre-graph metadata
- build similarity kNN edges + MST for connectivity
- personalized PageRank with teleport from node priors
- budgeted submodular selection (no hard caps)
- quality gates degrade confidence but never disable graph
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

from spectrue_core.utils.trace import Trace

from .candidates import build_knn_edges, mst_connectivity
from .quality_gates import confidence_from_density
from .ranking import compute_pagerank_with_ranks
from .selection import greedy_budgeted_submodular
from .types import (
    ClaimNode,
    ClaimPostGraphMeta,
    EdgeRelation,
    GraphResult,
    STRUCTURAL_RELATIONS,
    CandidateEdge,
)
from .deduplication import deduplicate_claims
from .edge_typing import type_edges
from .metadata import compute_pre_metadata, build_costs, build_ranked
from .tracing import (
    trace_pre_metadata,
    trace_knn,
    trace_edges,
    trace_mst,
    trace_pagerank,
    trace_selection,
    trace_quality,
)


if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from spectrue_core.runtime_config import ClaimGraphConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding clients to avoid domain -> adapter dependency."""
    async def embed_texts(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        ...

    def build_similarity_matrix(self, embeddings: list[list[float]]) -> list[list[float]]:
        ...


class ClaimGraphBuilder:
    """
    Deterministic ClaimGraph builder (no heuristics, no hard caps).
    """

    def __init__(
        self,
        config: "ClaimGraphConfig",
        openai_client: "AsyncOpenAI | None" = None,
        edge_typing_skill: object | None = None,  # kept for interface compatibility
        embedding_client: Embedder | None = None,
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.edge_typing_skill = edge_typing_skill

    async def build(self, claims: list[dict]) -> GraphResult:
        """
        Build claim graph with connectivity guarantees, personalized PageRank,
        and budgeted selection.
        """
        start_time = time.time()
        result = GraphResult(claims_count_raw=len(claims))

        if not claims:
            return result

        if not self.embedding_client:
             logger.warning("ClaimGraphBuilder: No embedding client provided.")
             return result

        try:
            position_map = {str(c.get("id") or f"c{i+1}"): i + 1 for i, c in enumerate(claims)}
            nodes = [ClaimNode.from_claim_dict(c, i) for i, c in enumerate(claims)]

            dedup = deduplicate_claims(nodes)
            nodes = dedup.canonical_claims
            result.claims_count_dedup = len(nodes)

            if not nodes:
                return result

            texts = [n.text for n in nodes]
            embeddings = await self.embedding_client.embed_texts(texts, purpose="query")
            sim_matrix = self.embedding_client.build_similarity_matrix(embeddings)

            # kNN edges + MST
            knn_edges, knn_map = build_knn_edges(
                nodes=nodes, similarity_matrix=sim_matrix, k=self.config.k_sim
            )
            mst_edges = mst_connectivity(
                node_ids=[n.claim_id for n in nodes],
                candidate_edges=knn_edges,
                similarity_matrix=sim_matrix,
                max_nodes_for_full_pairwise=self.config.max_nodes_for_full_pairwise,
            )
            # Build candidate set from kNN + MST using RAW cosine similarity.
            # Positional decay is applied AFTER edge typing for PageRank weighting
            # so the LLM sees the true semantic similarity when classifying.
            edge_set: dict[tuple[str, str], tuple[str, str, float]] = {
                tuple(sorted((u, v))): (u, v, w) for u, v, w in knn_edges
            }
            for u, v, w in mst_edges:
                edge_set.setdefault(tuple(sorted((u, v))), (u, v, w))
            sim_edges = list(edge_set.values())
            result.candidate_edges_count = len(sim_edges)
            result.sim_edges = sim_edges
            result.mst_edges = mst_edges

            # Pre-graph metadata + traces
            pre_meta = compute_pre_metadata(
                config=self.config,
                nodes=nodes,
                knn_map=knn_map,
                position_map=position_map,
            )
            result.pre_meta = pre_meta
            trace_pre_metadata(pre_meta)
            trace_knn(knn_map, self.config.trace_top_k)
            trace_edges(sim_edges, self.config.trace_top_k)
            trace_mst(mst_edges, len(nodes), self.config.trace_top_k)

            kept_edges, typed_edges_by_relation, structural_in, contradict_in = await type_edges(
                edge_typing_skill=self.edge_typing_skill,
                sim_edges=sim_edges,
                nodes=nodes,
                min_edge_score=0.65,
            )

            # Apply positional decay to kept edge scores for downstream PageRank
            for edge in kept_edges:
                pos_u = position_map.get(edge.src_id, 1)
                pos_v = position_map.get(edge.dst_id, 1)
                decay = math.exp(-abs(pos_u - pos_v) / max(self.config.edge_pos_gamma, 1e-6))
                edge.score = float(edge.score) * decay

            result.typed_edges = kept_edges
            result.typed_edges_kept_count = len(kept_edges)
            result.typed_edges_by_relation = typed_edges_by_relation
            result.within_topic_edges_count = sum(
                1 for e in kept_edges if not getattr(e, "cross_topic", False)
            )
            result.cross_topic_edges_count = sum(
                1 for e in kept_edges if getattr(e, "cross_topic", False)
            )

            # Personalized PageRank
            teleport = {cid: meta.node_prior for cid, meta in pre_meta.items()}
            node_ids = [n.claim_id for n in nodes]
            pr_scores, centrality_rank = compute_pagerank_with_ranks(
                node_ids=node_ids,
                edges=sim_edges,
                teleport=teleport,
                alpha=self.config.pagerank_alpha,
                eps=self.config.pagerank_eps,
                max_iter=self.config.pagerank_max_iter,
            )

            post_meta: dict[str, ClaimPostGraphMeta] = {}
            for cid in node_ids:
                post_meta[cid] = ClaimPostGraphMeta(
                    pagerank=pr_scores.get(cid, 0.0),
                    centrality_rank=centrality_rank.get(cid, -1),
                )
            result.post_meta = post_meta

            # Selection (budgeted submodular)
            id_to_idx = {n.claim_id: i for i, n in enumerate(nodes)}

            def sim_fn(a: str, b: str) -> float:
                ia = id_to_idx.get(a)
                ib = id_to_idx.get(b)
                if ia is None or ib is None:
                    return 0.0
                return float(sim_matrix[ia][ib])

            cost_map, cost_info = build_costs(
                claims, default=self.config.default_claim_cost
            )
            fallback_cost = max(float(self.config.default_claim_cost or 0.0), 1.0)
            budget = float(self.config.selection_budget or 0.0)
            if budget <= 0:
                # Deterministic finite budget to avoid rank-only/unlimited mode.
                target_count = self.config.top_k or len(node_ids) or 1
                budget = fallback_cost * float(target_count)
            selection_mode = "budgeted"
            selected: list[str] = []
            selection_trace_logs: list[dict] = []

            missing_all_costs = cost_info.get("missing_costs", 0) >= len(node_ids)
            if missing_all_costs:
                fallback_k = self.config.top_k or len(node_ids) or 0
                selected = sorted(node_ids, key=lambda x: pr_scores.get(x, 0.0), reverse=True)[
                    :fallback_k
                ]
                selection_mode = "rank_only_missing_costs"
                selection_trace_logs = [
                    {
                        "id": cid,
                        "gain": pr_scores.get(cid, 0.0),
                        "cost": 0.0,
                        "remaining_budget": budget,
                    }
                    for cid in selected
                ]
            elif budget > 0 and cost_map:
                selected, selection_trace_logs = greedy_budgeted_submodular(
                    nodes=node_ids,
                    sim=sim_fn,
                    pagerank=pr_scores,
                    cost=cost_map,
                    budget=budget,
                    lambda_rank=self.config.lambda_rank,
                    mu_redundancy=self.config.mu_redundancy,
                )
                selection_mode = "budgeted"

            if not selected:
                # Fail-open: fallback to top-K by PageRank when costs missing/invalid.
                fallback_k = self.config.top_k or len(node_ids) or 0
                selected = sorted(node_ids, key=lambda x: pr_scores.get(x, 0.0), reverse=True)[
                    :fallback_k
                ]
                selection_mode = "fallback_rank"
                selection_trace_logs = [
                    {
                        "id": cid,
                        "gain": pr_scores.get(cid, 0.0),
                        "cost": 0.0,
                        "remaining_budget": budget,
                    }
                    for cid in selected
                ]

            result.selection_trace = selection_trace_logs
            for step in selection_trace_logs:
                cid = step["id"]
                meta = post_meta.get(cid)
                if meta:
                    meta.selected = True
                    meta.selection_gain = step["gain"]
                    meta.selection_cost = step["cost"]
                    meta.debug = {
                        "remaining_budget": step["remaining_budget"],
                        "mode": selection_mode,
                        "cost_source": cost_info.get("source"),
                        "missing_costs": cost_info.get("missing_costs", 0),
                        "invalid_costs": cost_info.get("invalid_costs", 0),
                    }
            trace_selection(selected, selection_trace_logs, budget, cost_info, selection_mode, self.config.trace_top_k)

            ranked = build_ranked(
                node_ids,
                pr_scores,
                selected,
                structural_in=structural_in,
                contradict_in=contradict_in,
            )
            result.all_ranked = ranked
            result.key_claims = [r for r in ranked if r.is_key_claim]

            # Quality gate (degrade only)
            conf_scalar, quality_info = confidence_from_density(
                num_candidates=len(knn_edges),
                num_edges=len(sim_edges),
                min_kept_ratio=self.config.min_kept_ratio,
                max_kept_ratio=self.config.max_kept_ratio,
                beta_prior_alpha=self.config.beta_prior_alpha,
                beta_prior_beta=self.config.beta_prior_beta,
                result=result,
            )
            result.confidence_scalar = conf_scalar
            result.kept_ratio = result.kept_ratio_within_topic
            trace_quality(conf_scalar, quality_info)

            # Pagerank trace after quality to include priors
            degree_weights: dict[str, float] = defaultdict(float)
            for u, v, w in sim_edges:
                degree_weights[u] += w
                degree_weights[v] += w
            trace_pagerank(pr_scores, pre_meta, centrality_rank, degree_weights, self.config.trace_top_k)

            elapsed_ms = int((time.time() - start_time) * 1000)
            result.latency_ms = elapsed_ms
            logger.debug(
                "ClaimGraph complete: %d claims, %d edges, %d selected (%.1fms)",
                len(nodes),
                len(sim_edges),
                len(selected),
                elapsed_ms,
            )
            return result
        except Exception as e:
            logger.warning("ClaimGraph failed: %s", e)
            Trace.event("claim_graph.error", {"error": str(e)[:200]})
            result.disabled = False
            return result
