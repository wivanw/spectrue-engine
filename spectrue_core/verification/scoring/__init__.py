"""Scoring and aggregation modules."""

from .rgba_aggregation import apply_dependency_penalties, apply_conflict_explainability_penalty

__all__ = [
    "apply_dependency_penalties",
    "apply_conflict_explainability_penalty",
]