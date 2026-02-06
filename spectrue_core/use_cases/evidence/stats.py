"""Evidence stats use cases."""

from __future__ import annotations

from typing import Any, Callable

from spectrue_core.domain.evidence.stats import compute_evidence_stats_by_claim


def compute_stats(*, sources: list[dict], claims: list[dict], cluster_map: dict[str, str], cluster_sufficiency: dict[str, Any], normalize_url: Callable[[str], str], evidence_by_claim: dict | None = None):
    return compute_evidence_stats_by_claim(
        sources=sources,
        claims=claims,
        cluster_map=cluster_map,
        cluster_sufficiency=cluster_sufficiency,
        normalize_url=normalize_url,
        evidence_by_claim=evidence_by_claim,
    )
