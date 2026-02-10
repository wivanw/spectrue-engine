"""Evidence corroboration use cases."""

from __future__ import annotations

from spectrue_core.domain.evidence.corroboration import compute_corroboration_by_claim


def compute_corroboration(*, sources: list[dict], claims: list[dict], evidence_by_claim: dict | None = None):
    return compute_corroboration_by_claim(
        sources=sources,
        claims=claims,
        evidence_by_claim=evidence_by_claim,
    )
