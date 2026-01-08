# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Quality Gates

Quality gates degrade confidence instead of disabling the graph.
"""

from __future__ import annotations

import logging

from spectrue_core.graph.types import GraphResult

logger = logging.getLogger(__name__)


def confidence_from_density(
    *,
    num_candidates: int,
    num_edges: int,
    min_kept_ratio: float,
    max_kept_ratio: float,
    beta_prior_alpha: float,
    beta_prior_beta: float,
    result: GraphResult,
) -> tuple[float, dict]:
    """
    Compute a confidence scalar based on kept edge density (Beta posterior mean).

    Returns:
        (confidence_scalar, info_dict)
    """
    a = max(beta_prior_alpha, 1e-6)
    b = max(beta_prior_beta, 1e-6)
    k = max(0, num_edges)
    n = max(0, num_candidates)
    if k > n:
        logger.warning("quality_gate: num_edges %d > num_candidates %d, clamping", k, n)
        k = n

    kept_ratio = k / max(1, n)
    result.kept_ratio_within_topic = kept_ratio

    scalar = (a + k) / (a + b + n)
    reason = "posterior_mean"

    if kept_ratio < min_kept_ratio:
        result.sparse_graph = True
        reason = "below_min_density"
    elif kept_ratio > max_kept_ratio:
        reason = "above_max_density"

    return scalar, {
        "kept_ratio": kept_ratio,
        "reason": reason,
        "posterior_alpha": a + k,
        "posterior_beta": b + n - k,
        "min_kept_ratio": min_kept_ratio,
        "max_kept_ratio": max_kept_ratio,
    }
