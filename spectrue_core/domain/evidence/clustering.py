"""Evidence clustering helpers."""

from __future__ import annotations

from typing import Any


def group_by_similar_cluster_id(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group evidence items by their precomputed similar_cluster_id."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("similar_cluster_id") or "")
        if not cid:
            continue
        clusters.setdefault(cid, []).append(item)
    return clusters
