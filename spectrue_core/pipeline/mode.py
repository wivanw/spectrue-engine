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

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from spectrue_core.verification.types import SearchDepth


class AnalysisMode(str, Enum):
    """API-facing analysis mode names.
    
    This is the single source of truth for analysis_mode values
    in API responses. Maps internal pipeline mode names to
    frontend-compatible names.
    
    Mapping:
        - internal "general" → API "general"
        - internal "deep" → API "deep"
        - internal "deep_v2" → API "deep_v2"
    """
    GENERAL = "general"  # Standard single-claim analysis
    DEEP = "deep"        # Multi-claim per-claim RGBA
    DEEP_V2 = "deep_v2"  # Clustered retrieval + evidence stats

    def __str__(self) -> str:
        return self.value


class ScoringMode(str, Enum):
    """Scoring validation modes."""
    STANDARD = "standard"  # Full validation and clamping
    DEEP = "deep"          # Per-claim judging, minimal validation


@dataclass(frozen=True)
class PipelineMode:
    """
    Frozen configuration for a pipeline mode.

    This is the single source of truth for mode invariants.
    All mode-specific logic should consult these flags instead
    of checking string mode names.

    Attributes:
        name: Mode name ("general", "deep", or "deep_v2")
        allow_batch: Whether batch claim processing is allowed
        allow_clustering: Whether stance clustering is enabled
        require_single_language: Whether input must be single-language
        require_metering: Whether cost metering is required
        max_claims_for_scoring: Maximum number of claims to score (0 = unlimited)
        search_depth: Default search depth ("basic" or "advanced")
    """

    name: Literal["general", "deep", "deep_v2"]
    allow_batch: bool
    allow_clustering: bool
    require_single_language: bool
    require_metering: bool
    max_claims_for_scoring: int
    search_depth: Literal["basic", "advanced"]

    def __str__(self) -> str:
        return f"PipelineMode({self.name})"

    def __repr__(self) -> str:
        return (
            f"PipelineMode(name={self.name!r}, allow_batch={self.allow_batch}, "
            f"allow_clustering={self.allow_clustering}, "
            f"require_single_language={self.require_single_language}, "
            f"max_claims={self.max_claims_for_scoring}, "
            f"search_depth={self.search_depth!r})"
        )

    @property
    def api_analysis_mode(self) -> AnalysisMode:
        """Get API-facing analysis mode name.
        
        Maps internal mode name to frontend-compatible AnalysisMode enum.
        Use this for all API responses instead of raw mode.name.
        """
        try:
            return AnalysisMode(self.name)
        except ValueError:
            return AnalysisMode.GENERAL

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
