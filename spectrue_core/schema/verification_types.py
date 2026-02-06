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
Verification Types (Canonical)

This module centralizes common types used across the verification pipeline.
It exists to prevent "shape drift" (e.g. different call sites assuming
different source/search result formats).
"""

from __future__ import annotations

from typing import Any, TypedDict
from enum import Enum


class SearchProfileName(str, Enum):
    """Search policy profile names."""
    GENERAL = "general"  # Single claim, basic search
    DEEP = "deep"        # Multi-claim, intensive search


class SearchDepth(str, Enum):
    """Search depth levels for retrieval."""
    BASIC = "basic"        # Fast, limited results
    ADVANCED = "advanced"  # Thorough, more results


class StancePassMode(str, Enum):
    """Stance detection pass modes."""
    SINGLE = "single"      # Single pass (faster)
    TWO_PASS = "two_pass"  # Two-pass refinement (more accurate)


class Source(TypedDict, total=False):
    """
    Canonical evidence Source shape used across the engine.

    NOTE: Providers may emit different keys; normalization should map:
    - `link` -> `url`
    - `snippet` -> `content`
    """

    url: str
    title: str
    content: str
    snippet: str
    quote: str
    stance: str
    domain: str
    claim_id: str
    source_type: str
    is_trusted: bool
    relevance_score: float
    score: float


SearchResponse = tuple[str, list[Source]]
"""
Canonical response type for search operations:
    (context_text, sources)
"""


JsonDict = dict[str, Any]

