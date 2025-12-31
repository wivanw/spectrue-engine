# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Pipeline Mode

Defines the single source of truth for pipeline mode invariants.
All mode-specific behavior decisions should reference PipelineMode,
not scattered if-statements.

Usage:
    from spectrue_core.pipeline.mode import PipelineMode, NORMAL_MODE, DEEP_MODE

    mode = NORMAL_MODE
    if mode.allow_batch:
        # handle batch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineMode:
    """
    Frozen configuration for a pipeline mode.

    This is the single source of truth for mode invariants.
    All mode-specific logic should consult these flags instead
    of checking string mode names.

    Attributes:
        name: Mode name ("normal" or "deep")
        allow_batch: Whether batch claim processing is allowed
        allow_clustering: Whether stance clustering is enabled
        require_single_language: Whether input must be single-language
        require_metering: Whether cost metering is required
        max_claims_for_scoring: Maximum number of claims to score (0 = unlimited)
        search_depth: Default search depth ("basic" or "advanced")
    """

    name: Literal["normal", "deep"]
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


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Mode Instances
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_MODE = PipelineMode(
    name="normal",
    allow_batch=False,
    allow_clustering=False,
    require_single_language=True,
    require_metering=True,
    max_claims_for_scoring=1,
    search_depth="basic",
)
"""
Normal mode: Single claim, single language, no clustering.

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
    search_depth="advanced",
)
"""
Deep mode: Batch claims, multi-language, clustering enabled.

Use for comprehensive verification where all claims are processed
with advanced search and stance clustering.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mode Registry
# ─────────────────────────────────────────────────────────────────────────────

_MODE_REGISTRY: dict[str, PipelineMode] = {
    "normal": NORMAL_MODE,
    "general": NORMAL_MODE,  # alias
    "deep": DEEP_MODE,
}


def get_mode(name: str) -> PipelineMode:
    """
    Get a PipelineMode by name.

    Args:
        name: Mode name ("normal", "general", "deep")

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
