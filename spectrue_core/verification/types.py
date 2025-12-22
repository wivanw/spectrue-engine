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

