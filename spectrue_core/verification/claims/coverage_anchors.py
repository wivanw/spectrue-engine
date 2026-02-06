"""Backward-compatible exports for coverage anchors."""

from spectrue_core.domain.claims.extraction import (  # noqa: F401
    Anchor,
    AnchorKind,
    TIME_PATTERNS,
    NUMERIC_PATTERNS,
    QUOTE_PATTERNS,
    extract_all_anchors,
    extract_time_anchors,
    extract_numeric_anchors,
    extract_quote_anchors,
    get_anchor_ids,
    anchors_to_prompt_context,
)

__all__ = [
    "Anchor",
    "AnchorKind",
    "TIME_PATTERNS",
    "NUMERIC_PATTERNS",
    "QUOTE_PATTERNS",
    "extract_all_anchors",
    "extract_time_anchors",
    "extract_numeric_anchors",
    "extract_quote_anchors",
    "get_anchor_ids",
    "anchors_to_prompt_context",
]
