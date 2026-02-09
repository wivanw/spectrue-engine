from __future__ import annotations

from typing import Any, TypedDict
from spectrue_core.domain.verification.verdict.model import (
    SearchProfileName,
    SearchDepth,
    StancePassMode,
)

__all__ = [
    "SearchProfileName",
    "SearchDepth",
    "StancePassMode",
    "Source",
    "SearchResponse",
    "JsonDict",
]

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
