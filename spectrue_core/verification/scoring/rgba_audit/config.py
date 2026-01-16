# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Configuration defaults for RGBA audit aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RGBAAuditConfig:
    # Redundancy clustering
    redundancy_jaccard_threshold: float = 0.85
    redundancy_k: int = 3

    # Evidence aggregation thresholds
    conflict_threshold: float = 0.35
    c0: float = 0.15
    c1: float = 0.85

    # Core weight tables
    stance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "support": 1.0,
            "refute": 1.0,
            "unclear": 0.35,
            "unrelated": 0.0,
        }
    )
    directness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "direct": 1.0,
            "indirect": 0.7,
            "tangential": 0.4,
        }
    )
    specificity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "high": 1.0,
            "medium": 0.75,
            "low": 0.55,
        }
    )
    quote_integrity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ok": 1.0,
            "partial": 0.7,
            "out_of_context": 0.3,
            "not_applicable": 0.5,
        }
    )
    novelty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "original": 1.0,
            "syndicated": 0.65,
            "unknown": 0.8,
        }
    )

    # Source reliability priors (tier-based)
    source_reliability_priors: dict[str, float] = field(
        default_factory=lambda: {
            "A": 0.9,
            "B": 0.75,
            "C": 0.55,
            "D": 0.35,
            "unknown": 0.5,
        }
    )

    # Assertion strength expectations for honesty (B)
    assertion_strength_weights: dict[str, float] = field(
        default_factory=lambda: {
            "weak": 0.35,
            "medium": 0.55,
            "strong": 0.75,
        }
    )

    # Metric weights (A/B/R)
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "A": 1.0,
            "B": 1.0,
            "R": 1.0,
        }
    )

    # Auditability and risk tuning
    auditability_weight: float = 0.9
    honesty_weight: float = 0.9
    risk_weight: float = 0.8


def default_rgba_audit_config() -> RGBAAuditConfig:
    return RGBAAuditConfig()
