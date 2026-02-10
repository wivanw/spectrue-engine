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
Pipeline Mode

Defines the single source of truth for pipeline mode invariants.
All mode-specific behavior decisions should reference PipelineMode,
not scattered if-statements.

Usage:
    from spectrue_core.pipeline.mode import PipelineMode, NORMAL_MODE, DEEP_MODE, DEEP_V2_MODE
    from spectrue_core.pipeline.mode import AnalysisMode

    mode = NORMAL_MODE
    if mode.allow_batch:
        # handle batch

    # Get API-facing analysis mode name
    api_mode = mode.api_analysis_mode  # Returns AnalysisMode.GENERAL
"""

from __future__ import annotations

from spectrue_core.use_cases.types import PipelineMode, SearchDepth, AnalysisMode, ScoringMode


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Mode Instances
# ─────────────────────────────────────────────────────────────────────────────

GENERAL_MODE = PipelineMode(
    name="general",  # Renamed from 'normal' to match AnalysisMode
    allow_batch=False,
    allow_clustering=False,
    require_single_language=True,
    require_metering=True,
    max_claims_for_scoring=1,
    search_depth=SearchDepth.BASIC.value,
)
"""
General mode: Single claim, single language, no clustering.

Use for standard fact-checking requests where a single primary
claim is verified with basic search depth.
"""

DEEP_MODE = PipelineMode(
    name="deep",
    allow_batch=True,
    allow_clustering=True,
    require_single_language=False,
    require_metering=True,
    max_claims_for_scoring=0,  # unlimited
    search_depth=SearchDepth.ADVANCED.value,
)
"""
Deep mode: Batch claims, multi-language, clustering enabled.

Use for comprehensive verification where all claims are processed
with advanced search and stance clustering.
"""

DEEP_V2_MODE = PipelineMode(
    name="deep_v2",
    allow_batch=True,
    allow_clustering=True,
    require_single_language=False,
    require_metering=True,
    max_claims_for_scoring=0,  # unlimited
    search_depth=SearchDepth.ADVANCED.value,
)
"""
Deep v2 mode: Batch claims, multi-language, clustered retrieval enabled.

Use for comprehensive verification with claim clustering and per-claim judging.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mode Registry
# ─────────────────────────────────────────────────────────────────────────────

_MODE_REGISTRY: dict[str, PipelineMode] = {
    "general": GENERAL_MODE,
    "deep": DEEP_MODE,
    "deep_v2": DEEP_V2_MODE,
}


def get_mode(name: str) -> PipelineMode:
    """
    Get a PipelineMode by name.

    Args:
        name: Mode name ("general", "deep", "deep_v2")

    Returns:
        The corresponding PipelineMode instance

    Raises:
        ValueError: If mode name is not recognized
    """
    normalized = name.lower().strip()
    if normalized not in _MODE_REGISTRY:
        valid = ", ".join(sorted(_MODE_REGISTRY.keys()))
        raise ValueError(f"Unknown pipeline mode: {name!r}. Valid modes: {valid}")
    return _MODE_REGISTRY[normalized]
