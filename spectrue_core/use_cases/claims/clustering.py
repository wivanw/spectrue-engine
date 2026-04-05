"""Claim clustering use cases."""

from __future__ import annotations

from spectrue_core.domain.claims.clustering import build_soft_clusters_from_graph_result, build_claim_clusters


def build_soft_clusters(*, claims: list[dict], graph_result):
    return build_soft_clusters_from_graph_result(claims=claims, graph_result=graph_result)


def build_clusters(*, claims: list[dict], graph_result, quantile: float, representative_min_k: int, representative_max_k: int):
    return build_claim_clusters(
        claims=claims,
        graph_result=graph_result,
        quantile=quantile,
        representative_min_k=representative_min_k,
        representative_max_k=representative_max_k,
    )
