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
from typing import TYPE_CHECKING

from spectrue_core.graph.candidates import build_knn_edges, mst_connectivity
from spectrue_core.graph.embedding_util import EmbeddingClient
from spectrue_core.graph.quality_gates import confidence_from_density
from spectrue_core.graph.ranking import compute_pagerank_with_ranks
from spectrue_core.graph.selection import greedy_budgeted_submodular
from spectrue_core.graph.types import (
    ClaimNode,
    ClaimPostGraphMeta,
    ClaimPreGraphMeta,
    DedupeResult,
    EdgeRelation,
    GraphResult,
    RankedClaim,
    STRUCTURAL_RELATIONS,
    CandidateEdge,
)
from spectrue_core.utils.trace import Trace

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from spectrue_core.runtime_config import ClaimGraphConfig

logger = logging.getLogger(__name__)


class ClaimGraphBuilder:
    """
    Deterministic ClaimGraph builder (no heuristics, no hard caps).
    """

    def __init__(
        self,
        config: "ClaimGraphConfig",
        openai_client: "AsyncOpenAI | None" = None,
        edge_typing_skill: object | None = None,  # kept for interface compatibility
        embedding_client: EmbeddingClient | None = None,
    ):
        self.config = config
        self.embedding_client = embedding_client or EmbeddingClient(openai_client)
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

        try:
            position_map = {str(c.get("id") or f"c{i+1}"): i + 1 for i, c in enumerate(claims)}
            nodes = [ClaimNode.from_claim_dict(c, i) for i, c in enumerate(claims)]

            dedup = self._deduplicate(nodes)
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
            # Apply positional decay only for weighting (not for MST connectivity)
            pos_adj_edges: list[tuple[str, str, float]] = []
            for u, v, sim in knn_edges:
                pos_u = position_map.get(u, 1)
                pos_v = position_map.get(v, 1)
                decay = math.exp(-abs(pos_u - pos_v) / max(self.config.edge_pos_gamma, 1e-6))
                pos_adj_edges.append((u, v, sim * decay))
            edge_set = {tuple(sorted((u, v))): (u, v, w) for u, v, w in pos_adj_edges}
            for u, v, w in mst_edges:
                edge_set.setdefault(tuple(sorted((u, v))), (u, v, w))
            sim_edges = list(edge_set.values())
            result.candidate_edges_count = len(sim_edges)
            result.sim_edges = sim_edges
            result.mst_edges = mst_edges

            # Pre-graph metadata + traces
            pre_meta = self._compute_pre_metadata(
                nodes=nodes,
                knn_map=knn_map,
                position_map=position_map,
            )
            result.pre_meta = pre_meta
            self._trace_pre_metadata(pre_meta)
            self._trace_knn(knn_map)
            self._trace_edges(sim_edges)
            self._trace_mst(mst_edges, len(nodes))

            typed_edges: list[object] = []
            typed_edges_by_relation: dict[str, int] = {}
            structural_in: dict[str, float] = {}
            contradict_in: dict[str, float] = {}

            if self.edge_typing_skill and sim_edges:
                node_map = {n.claim_id: n for n in nodes}
                candidates: list[CandidateEdge] = []
                for src, dst, sim_score in sim_edges:
                    src_node = node_map.get(src)
                    dst_node = node_map.get(dst)
                    same_section = (
                        src_node.section_id == dst_node.section_id if src_node and dst_node else False
                    )
                    candidates.append(
                        CandidateEdge(
                            src_id=src,
                            dst_id=dst,
                            reason="sim",
                            sim_score=float(sim_score),
                            same_section=same_section,
                            cross_topic=False,
                        )
                    )

                try:
                    typed_edges = await self.edge_typing_skill.type_edges_batch(
                        candidates, node_map
                    )
                except Exception as exc:
                    logger.warning("[M72] Edge typing failed: %s", exc)
                    Trace.event("edge_typing.error", {"error": str(exc)[:200]})
                    typed_edges = []

            kept_edges = []
            min_edge_score = 0.6
            for te in typed_edges or []:
                if not te or te.relation == EdgeRelation.UNRELATED:
                    continue
                if float(te.score) < min_edge_score:
                    continue
                kept_edges.append(te)
                typed_edges_by_relation[te.relation.value] = (
                    typed_edges_by_relation.get(te.relation.value, 0) + 1
                )
                if te.relation in STRUCTURAL_RELATIONS:
                    structural_in[te.dst_id] = structural_in.get(te.dst_id, 0.0) + float(te.score)
                if te.relation == EdgeRelation.CONTRADICTS:
                    contradict_in[te.dst_id] = contradict_in.get(te.dst_id, 0.0) + float(te.score)

            result.typed_edges = kept_edges
            result.typed_edges_kept_count = len(kept_edges)
            result.typed_edges_by_relation = typed_edges_by_relation

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

            def sim(a: str, b: str) -> float:
                ia = id_to_idx.get(a)
                ib = id_to_idx.get(b)
                if ia is None or ib is None:
                    return 0.0
                return float(sim_matrix[ia][ib])

            cost_map, cost_info = self._build_costs(
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
            selection_trace: list[dict] = []

            missing_all_costs = cost_info.get("missing_costs", 0) >= len(node_ids)
            if missing_all_costs:
                fallback_k = self.config.top_k or len(node_ids) or 0
                selected = sorted(node_ids, key=lambda x: pr_scores.get(x, 0.0), reverse=True)[
                    :fallback_k
                ]
                selection_mode = "rank_only_missing_costs"
                selection_trace = [
                    {
                        "id": cid,
                        "gain": pr_scores.get(cid, 0.0),
                        "cost": 0.0,
                        "remaining_budget": budget,
                    }
                    for cid in selected
                ]
            elif budget > 0 and cost_map:
                selected, selection_trace = greedy_budgeted_submodular(
                    nodes=node_ids,
                    sim=sim,
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
                selection_trace = [
                    {
                        "id": cid,
                        "gain": pr_scores.get(cid, 0.0),
                        "cost": 0.0,
                        "remaining_budget": budget,
                    }
                    for cid in selected
                ]

            result.selection_trace = selection_trace
            for step in selection_trace:
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
            self._trace_selection(selected, selection_trace, budget, cost_info, selection_mode)

            ranked = self._build_ranked(
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
            self._trace_quality(conf_scalar, quality_info)

            # Pagerank trace after quality to include priors
            degree_weights: dict[str, float] = defaultdict(float)
            for u, v, w in sim_edges:
                degree_weights[u] += w
                degree_weights[v] += w
            self._trace_pagerank(pr_scores, pre_meta, centrality_rank, degree_weights)

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

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _compute_pre_metadata(
        self,
        *,
        nodes: list[ClaimNode],
        knn_map: dict[str, list[tuple[str, float]]],
        position_map: dict[str, int],
    ) -> dict[str, ClaimPreGraphMeta]:
        """Compute pre-graph priors from kNN similarities."""
        pre_meta: dict[str, ClaimPreGraphMeta] = {}
        gamma = float(self.config.pos_prior_gamma)
        w_pos = float(self.config.w_pos)
        w_supp = float(self.config.w_supp)
        w_imp = float(self.config.w_imp)
        w_harm = float(self.config.w_harm)

        importance_lookup = {n.claim_id: n.importance for n in nodes}
        harm_lookup = {n.claim_id: n.harm_potential for n in nodes}

        for node in nodes:
            neighbors = knn_map.get(node.claim_id, [])
            sims = [max(0.0, s) for _, s in neighbors]
            support_mass = sum(sims)
            novelty = 1.0 - max(sims) if sims else 1.0
            if support_mass > 0:
                probs = [s / support_mass for s in sims if s > 0]
                uncertainty = -sum(p * math.log(p) for p in probs if p > 0)
            else:
                uncertainty = 0.0

            pos_rank = position_map.get(node.claim_id, 1)
            pos_prior = math.exp(-gamma * float(pos_rank))
            importance_prior = float(importance_lookup.get(node.claim_id, 0.0))
            harm_prior = float(harm_lookup.get(node.claim_id, 0.0)) / 5.0
            node_prior = max(
                0.0,
                w_pos * pos_prior
                + w_supp * support_mass
                + w_imp * importance_prior
                + w_harm * harm_prior,
            )

            pre_meta[node.claim_id] = ClaimPreGraphMeta(
                claim_id=node.claim_id,
                position_rank=pos_rank,
                pos_prior=pos_prior,
                support_mass=support_mass,
                novelty=novelty,
                uncertainty_proxy=uncertainty,
                importance_prior=importance_prior,
                harm_prior=harm_prior,
                node_prior=node_prior,
            )
        return pre_meta

    def _build_costs(self, claims: list[dict], default: float) -> tuple[dict[str, float], dict]:
        """
        Build deterministic cost map with guaranteed coverage.
        Missing/invalid costs fall back to default (>=1).
        """
        cost_map: dict[str, float] = {}
        source = "provided"
        missing_costs = 0
        invalid_costs = 0
        fallback = max(float(default or 0.0), 1.0)
        for c in claims:
            cid = str(c.get("id") or "")
            if not cid:
                continue
            if "cost_estimate" in c:
                try:
                    val = float(c.get("cost_estimate") or 0.0)
                    if val > 0:
                        cost_map[cid] = val
                        continue
                    invalid_costs += 1
                except Exception:
                    invalid_costs += 1
            else:
                missing_costs += 1
            cost_map[cid] = fallback
            source = "default_claim_cost"

        return cost_map, {
            "source": source,
            "missing_costs": missing_costs,
            "invalid_costs": invalid_costs,
        }

    def _build_ranked(
        self,
        node_ids: list[str],
        pr_scores: dict[str, float],
        selected: list[str],
        *,
        structural_in: dict[str, float] | None = None,
        contradict_in: dict[str, float] | None = None,
    ) -> list[RankedClaim]:
        ranked: list[RankedClaim] = []
        structural_in = structural_in or {}
        contradict_in = contradict_in or {}
        for cid in sorted(node_ids, key=lambda x: pr_scores.get(x, 0.0), reverse=True):
            ranked.append(
                RankedClaim(
                    claim_id=cid,
                    centrality_score=pr_scores.get(cid, 0.0),
                    in_structural_weight=structural_in.get(cid, 0.0),
                    in_contradict_weight=contradict_in.get(cid, 0.0),
                    is_key_claim=cid in selected,
                )
            )
        return ranked

    # ------------------------------------------------------------------ #
    # Tracing helpers (compact payloads)
    # ------------------------------------------------------------------ #

    def _trace_pre_metadata(self, pre_meta: dict[str, ClaimPreGraphMeta]) -> None:
        payload = [
            {
                "id": m.claim_id,
                "pos_prior": round(m.pos_prior, 4),
                "support": round(m.support_mass, 4),
                "novelty": round(m.novelty, 4),
                "uncertainty": round(m.uncertainty_proxy, 4),
                "node_prior": round(m.node_prior, 4),
            }
            for m in pre_meta.values()
        ]
        Trace.event("claim_graph.pre_metadata", {"items": payload})

    def _trace_knn(self, knn_map: dict[str, list[tuple[str, float]]]) -> None:
        top_k = self.config.trace_top_k
        payload = {
            cid: [{"id": nid, "sim": round(sim, 4)} for nid, sim in neigh[:top_k]]
            for cid, neigh in knn_map.items()
        }
        Trace.event("claim_graph.knn", {"neighbors": payload})

    def _trace_edges(self, edges: list[tuple[str, str, float]]) -> None:
        if not edges:
            Trace.event("claim_graph.edges.sim", {"count": 0})
            return
        weights = [w for _, _, w in edges]
        mean_w = sum(weights) / len(weights)
        top = sorted(edges, key=lambda e: e[2], reverse=True)[: self.config.trace_top_k]
        Trace.event(
            "claim_graph.edges.sim",
            {
                "count": len(edges),
                "min": round(min(weights), 4),
                "max": round(max(weights), 4),
                "mean": round(mean_w, 4),
                "top": [(u, v, round(w, 4)) for u, v, w in top],
            },
        )

    def _trace_mst(self, mst_edges: list[tuple[str, str, float]], node_count: int) -> None:
        Trace.event(
            "claim_graph.mst",
            {
                "edges": [(u, v, round(w, 4)) for u, v, w in mst_edges[: self.config.trace_top_k]],
                "node_count": node_count,
                "connected": len(mst_edges) >= max(0, node_count - 1),
            },
        )

    def _trace_pagerank(
        self,
        pr_scores: dict[str, float],
        pre_meta: dict[str, ClaimPreGraphMeta],
        centrality_rank: dict[str, int],
        degree_weights: dict[str, float],
    ) -> None:
        top_ids = sorted(pr_scores, key=pr_scores.get, reverse=True)[: self.config.trace_top_k]
        Trace.event(
            "claim_graph.pagerank",
            {
                "top": [
                    {
                        "id": cid,
                        "pagerank": round(pr_scores.get(cid, 0.0), 6),
                        "prior": round(pre_meta.get(cid).node_prior, 6) if cid in pre_meta else 0.0,
                        "rank": centrality_rank.get(cid, -1),
                        "prior_components": {
                            "pos": round(pre_meta[cid].pos_prior, 6) if cid in pre_meta else 0.0,
                            "supp": round(pre_meta[cid].support_mass, 6) if cid in pre_meta else 0.0,
                            "imp": round(pre_meta[cid].importance_prior, 6) if cid in pre_meta else 0.0,
                            "harm": round(pre_meta[cid].harm_prior, 6) if cid in pre_meta else 0.0,
                        },
                        "degree_weight": round(degree_weights.get(cid, 0.0), 6),
                    }
                    for cid in top_ids
                ]
            },
        )

    def _trace_selection(self, selected: list[str], steps: list[dict], budget: float, cost_info: dict, mode: str) -> None:
        Trace.event(
            "claim_graph.selection",
            {
                "selected": selected,
                "budget": budget,
                "mode": mode,
                "cost_info": cost_info,
                "steps": [
                    {
                        "id": s["id"],
                        "gain": round(float(s["gain"]), 6),
                        "cost": round(float(s["cost"]), 6),
                        "remaining_budget": round(float(s["remaining_budget"]), 6),
                    }
                    for s in steps[: self.config.trace_top_k]
                ],
            },
        )

    def _trace_quality(self, scalar: float, info: dict) -> None:
        Trace.event(
            "claim_graph.quality",
            {
                "scalar": round(float(scalar), 4),
                **{k: v for k, v in info.items()},
            },
        )

    # ------------------------------------------------------------------ #
    # Dedup helpers (kept from earlier implementation)
    # ------------------------------------------------------------------ #

    def _deduplicate(self, nodes: list[ClaimNode]) -> DedupeResult:
        """
        Cluster near-duplicate claims using 90% Jaccard word overlap.
        """
        if not nodes:
            return DedupeResult([], {}, 1.0)

        groups: dict[str, list[ClaimNode]] = defaultdict(list)

        for node in nodes:
            key = " ".join(node.text.lower().split())
            groups[key].append(node)

        canonical_claims: list[ClaimNode] = []
        dedup_map: dict[str, list[str]] = {}

        for _, group in groups.items():
            if len(group) == 1:
                canonical_claims.append(group[0])
                continue

            group.sort(key=lambda n: n.importance, reverse=True)
            canonical = group[0]
            canonical_claims.append(canonical)

            merged_ids = [n.claim_id for n in group[1:]]
            if merged_ids:
                dedup_map[canonical.claim_id] = merged_ids

        canonical_claims = self._fuzzy_dedup(canonical_claims, threshold=0.9)
        reduction = len(nodes) / len(canonical_claims) if canonical_claims else 1.0

        return DedupeResult(canonical_claims, dedup_map, reduction)

    def _fuzzy_dedup(
        self,
        nodes: list[ClaimNode],
        threshold: float = 0.9,
    ) -> list[ClaimNode]:
        """Fuzzy deduplication using Jaccard word similarity."""
        if len(nodes) <= 1:
            return nodes

        sorted_nodes = sorted(nodes, key=lambda n: n.importance, reverse=True)
        kept: list[ClaimNode] = []

        for node in sorted_nodes:
            node_words = set(node.text.lower().split())
            if not node_words:
                continue

            is_duplicate = False
            for existing in kept:
                existing_words = set(existing.text.lower().split())
                if not existing_words:
                    continue

                intersection = len(node_words & existing_words)
                union = len(node_words | existing_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(node)

        return kept


def build_query_clusters(claims: list[dict]) -> dict[str, list[str]]:
    """
    Group claims into clusters for shared query planning.
    """
    clusters: dict[str, list[str]] = {}
    for idx, claim in enumerate(claims or []):
        claim_id = str(claim.get("id") or f"c{idx + 1}")
        cluster_key = (
            claim.get("topic_key")
            or claim.get("topic_group")
            or claim.get("id")
            or claim.get("claim_id")
            or "cluster_default"
        )
        clusters.setdefault(cluster_key, []).append(claim_id)
    return clusters
