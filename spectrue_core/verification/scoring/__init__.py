"""Scoring and aggregation modules."""

from .rgba_aggregation import apply_dependency_penalties, apply_conflict_explainability_penalty
from .rgba_audit import RGBAAuditConfig, aggregate_rgba_audit, default_rgba_audit_config

__all__ = [
    "apply_dependency_penalties",
    "apply_conflict_explainability_penalty",
    "RGBAAuditConfig",
    "default_rgba_audit_config",
    "aggregate_rgba_audit",
]
