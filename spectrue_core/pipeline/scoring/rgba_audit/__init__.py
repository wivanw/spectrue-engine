"""RGBA audit scoring and aggregation helpers."""

from .config import RGBAAuditConfig, default_rgba_audit_config
from .aggregation import aggregate_rgba_audit

__all__ = [
    "RGBAAuditConfig",
    "default_rgba_audit_config",
    "aggregate_rgba_audit",
]
