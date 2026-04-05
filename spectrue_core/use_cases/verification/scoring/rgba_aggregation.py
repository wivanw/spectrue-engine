"""Compatibility layer for RGBA aggregation."""

from spectrue_core.domain.verification.stance.aggregation import (
    ClaimScore,
    AggregatedRGBA,
    aggregate_weighted,
    claim_to_score,
    recompute_verified_score,
    apply_conflict_explainability_penalty,
    apply_dependency_penalties,
)

__all__ = [
    "ClaimScore",
    "AggregatedRGBA",
    "aggregate_weighted",
    "claim_to_score",
    "recompute_verified_score",
    "apply_conflict_explainability_penalty",
    "apply_dependency_penalties",
]
