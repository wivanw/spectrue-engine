"""Claim clustering helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from spectrue_core.domain.claims.graph.types import GraphResult
from spectrue_core.domain.claims.topics import normalize_entities
from spectrue_core.domain.verification.verdict.global_aggregation import is_unverifiable_by_nature


@dataclass(frozen=True)
class ClaimCluster:
    cluster_id: str
    claim_ids: tuple[str, ...]
    representative_claim_ids: tuple[str, ...]
    centrality_scores: dict[str, float]
    anchor_density_scores: dict[str, float]
    size: int
    # Entities of verifiable claims the representatives cover (set cover result).
    covered_entities: tuple[str, ...] = ()

    def to_summary(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "claim_ids": list(self.claim_ids),
            "representative_claim_ids": list(self.representative_claim_ids),
            "covered_entities": list(self.covered_entities),
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


def _entities(claim: dict[str, Any]) -> frozenset[str]:
    return normalize_entities(claim.get("subject_entities"))


def select_representatives(
    ranked_ids: list[str],
    claim_lookup: dict[str, dict[str, Any]],
    k_floor: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Representatives that cover every verifiable claim's subject entities.

    Cluster queries are built from representatives only, so a claim whose
    entities no representative carries gets no search at all. Greedy set cover
    over the entities of claims with verification_target != none: repeatedly
    take the claim that covers the most still-uncovered entities, ties broken by
    `ranked_ids` order (the existing importance score). Then top up to
    `k_floor` from `ranked_ids`. Coverage wins over any size cap.

    Returns (representative ids, covered entities).
    """
    verifiable = [cid for cid in ranked_ids if not is_unverifiable_by_nature(claim_lookup.get(cid, {}))]
    ents = {cid: _entities(claim_lookup.get(cid, {})) for cid in verifiable}
    uncovered = set().union(*ents.values()) if ents else set()
    covered = set(uncovered)
    chosen: list[str] = []
    while uncovered:
        best = max(
            (cid for cid in verifiable if cid not in chosen),
            key=lambda cid: (len(ents[cid] & uncovered), -ranked_ids.index(cid)),
        )
        if not ents[best] & uncovered:
            break
        chosen.append(best)
        uncovered -= ents[best]
    pool = verifiable or ranked_ids
    for cid in pool:
        if len(chosen) >= k_floor:
            break
        if cid not in chosen:
            chosen.append(cid)
    return tuple(chosen), tuple(sorted(covered))


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
    claim_lookup = {
        str(c.get("id") or c.get("claim_id") or f"c{i + 1}"): c
        for i, c in enumerate(claims)
    }

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
        representative_claim_ids, covered_entities = select_representatives(
            [cid for cid, _ in scored], claim_lookup, k
        )

        clusters.append(
            ClaimCluster(
                cluster_id=cluster_id,
                claim_ids=tuple(cluster_claims),
                representative_claim_ids=representative_claim_ids,
                centrality_scores={cid: float(centrality.get(cid, 0.0)) for cid in cluster_claims},
                anchor_density_scores={cid: float(anchor_density.get(cid, 0.0)) for cid in cluster_claims},
                size=len(cluster_claims),
                covered_entities=covered_entities,
            )
        )

    clusters.sort(key=lambda c: c.cluster_id)
    return clusters


def build_soft_clusters_from_graph_result(
    claims: list[dict[str, Any]],
    graph_result: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build soft clusters based on graph connected components."""
    if not graph_result:
        return [], {}

    claim_map = {str(c.get("id")): c for c in claims if c.get("id")}

    clusters: list[dict[str, Any]] = []
    for component_ids in graph_result.connected_components():
        component_claims = [claim_map[cid] for cid in component_ids if cid in claim_map]
        if not component_claims:
            continue

        shared_assertions = set()
        topic_tags = set()

        for claim in component_claims:
            assertions = claim.get("assertions", [])
            if isinstance(assertions, list):
                for a in assertions:
                    key = None
                    if hasattr(a, "key"):
                        key = a.key
                    elif isinstance(a, dict):
                        key = a.get("key")
                    if key:
                        shared_assertions.add(key)

            metadata = claim.get("metadata")
            if metadata:
                tags = (
                    getattr(metadata, "topic_tags", [])
                    if not isinstance(metadata, dict)
                    else metadata.get("topic_tags", [])
                )
                if isinstance(tags, (list, set)):
                    topic_tags.update(tags)

        clusters.append(
            {
                "cluster_id": f"cluster_{len(clusters)}",
                "claim_ids": [str(c.get("id")) for c in component_claims],
                "shared_assertion_keys": list(shared_assertions),
                "topic_tags": list(topic_tags),
            }
        )

    cluster_map: dict[str, str] = {}
    for c in clusters:
        for cid in c["claim_ids"]:
            cluster_map[cid] = c["cluster_id"]

    return clusters, cluster_map
