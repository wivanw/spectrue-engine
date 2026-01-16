# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from spectrue_core.graph.types import GraphResult


@dataclass(frozen=True)
class ClaimCluster:
    cluster_id: str
    claim_ids: tuple[str, ...]
    representative_claim_ids: tuple[str, ...]
    centrality_scores: dict[str, float]
    anchor_density_scores: dict[str, float]
    size: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "claim_ids": list(self.claim_ids),
            "representative_claim_ids": list(self.representative_claim_ids),
            "size": self.size,
        }


def _stable_cluster_id(claim_ids: list[str]) -> str:
    ordered = sorted([cid for cid in claim_ids if cid])
    raw = "|".join(ordered)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"clu_{digest}"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    idx = int(round(q * (len(ordered) - 1)))
    return float(ordered[idx])


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def _anchor_density_scores(claims: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    max_count = 0
    for claim in claims:
        claim_id = str(claim.get("id") or claim.get("claim_id") or "")
        if not claim_id:
            continue
        anchor_refs = claim.get("anchor_refs")
        count = len(anchor_refs) if isinstance(anchor_refs, list) else 0
        counts[claim_id] = count
        if count > max_count:
            max_count = count
    if max_count <= 0:
        return {cid: 0.0 for cid in counts}
    return {cid: count / max_count for cid, count in counts.items()}


def _centrality_scores(graph_result: GraphResult | None) -> dict[str, float]:
    if not graph_result:
        return {}
    scores: dict[str, float] = {}
    for ranked in graph_result.all_ranked or []:
        scores[str(ranked.claim_id)] = float(ranked.centrality_score)
    return scores


def _representative_k(cluster_size: int, min_k: int, max_k: int) -> int:
    min_k = max(1, int(min_k))
    max_k = max(min_k, int(max_k))
    if cluster_size <= min_k:
        return cluster_size
    if cluster_size <= max_k:
        return min_k
    return max_k


def _build_components(
    claim_ids: list[str],
    mst_edges: list[tuple[str, str, float]],
    *,
    quantile: float,
) -> list[list[str]]:
    if not claim_ids:
        return []
    if len(claim_ids) <= 1 or not mst_edges:
        return [[cid] for cid in claim_ids]

    sims = [float(edge[2]) for edge in mst_edges]
    tau = _quantile(sims, quantile)

    adjacency: dict[str, set[str]] = {cid: set() for cid in claim_ids}
    for u, v, w in mst_edges:
        if w >= tau:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

    visited: set[str] = set()
    components: list[list[str]] = []

    for cid in claim_ids:
        if cid in visited:
            continue
        queue = [cid]
        visited.add(cid)
        component: list[str] = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    return components


def build_claim_clusters(
    *,
    claims: list[dict[str, Any]],
    graph_result: GraphResult | None,
    quantile: float,
    representative_min_k: int,
    representative_max_k: int,
) -> list[ClaimCluster]:
    claim_ids = [
        str(c.get("id") or c.get("claim_id") or f"c{i + 1}")
        for i, c in enumerate(claims or [])
    ]
    if not claim_ids:
        return []

    mst_edges = []
    if graph_result:
        mst_edges = list(graph_result.mst_edges or [])

    components = _build_components(claim_ids, mst_edges, quantile=quantile)

    centrality = _centrality_scores(graph_result)
    anchor_density = _anchor_density_scores(claims)
    claim_lookup = {str(c.get("id") or c.get("claim_id") or f"c{i + 1}"): c for i, c in enumerate(claims)}

    clusters: list[ClaimCluster] = []
    for component in components:
        cluster_claims = [cid for cid in component if cid]
        cluster_claims.sort()
        cluster_id = _stable_cluster_id(cluster_claims)

        scored: list[tuple[str, float]] = []
        for cid in cluster_claims:
            claim = claim_lookup.get(cid, {})
            importance = _clamp01(claim.get("importance", 0.5), default=0.5)
            worthiness = _clamp01(
                claim.get("check_worthiness", claim.get("importance", 0.5)), default=0.5
            )
            centrality_term = max(0.0, centrality.get(cid, 0.0))
            anchor_term = max(0.0, anchor_density.get(cid, 0.0))
            score = importance * worthiness * (1.0 + centrality_term) * (1.0 + anchor_term)
            scored.append((cid, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        k = _representative_k(len(cluster_claims), representative_min_k, representative_max_k)
        representative_claim_ids = tuple([cid for cid, _ in scored[:k]])

        clusters.append(
            ClaimCluster(
                cluster_id=cluster_id,
                claim_ids=tuple(cluster_claims),
                representative_claim_ids=representative_claim_ids,
                centrality_scores={cid: float(centrality.get(cid, 0.0)) for cid in cluster_claims},
                anchor_density_scores={cid: float(anchor_density.get(cid, 0.0)) for cid in cluster_claims},
                size=len(cluster_claims),
            )
        )

    clusters.sort(key=lambda c: c.cluster_id)
    return clusters
