"""Claim clustering helpers for query planning."""
from __future__ import annotations


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
