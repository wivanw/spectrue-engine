# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Deterministic Coverage Anchors for Claim Extraction.

This module extracts deterministic anchors from raw text:
- Time anchors (dates in various formats)
- Numeric anchors (digit sequences with context)
- Quote anchors (quoted spans)

Anchors represent verifiable facts that must be accounted for during extraction.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from spectrue_core.utils.trace import Trace


class AnchorKind(Enum):
    """Anchor type classification."""
    TIME = "time"
    NUMBER = "number"
    QUOTE = "quote"


@dataclass
class Anchor:
    """
    A deterministic anchor extracted from text.
    
    Anchors are structural markers (dates, numbers, quotes) that represent
    verifiable facts which must be covered by extracted claims.
    """
    anchor_id: str          # Stable ID: "t1", "n2", "q1"
    kind: AnchorKind        # Type classification
    span_text: str          # Exact matched text
    char_start: int         # Start position in source text
    char_end: int           # End position in source text
    context_window: str     # ±5 tokens around anchor


# ================= Time Anchors =================

# Patterns for date detection (ordered by specificity)
TIME_PATTERNS = [
    # ISO format: 2024-01-15
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "iso_date"),
    # European: 15.01.2024, 15/01/2024
    (r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", "eu_date"),
    # American: 01/15/2024
    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "us_date"),
    # Year-month: 2024-01, January 2024
    (r"\b(\d{4})-(\d{2})\b", "year_month"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", "month_year"),
    # Quarter: Q1 2024, Q4 2023
    (r"\b[Qq]([1-4])\s*(\d{4})\b", "quarter"),
    # Standalone year: 2024, 2023 (but not 4-digit numbers in other contexts)
    (r"\b(19\d{2}|20\d{2})\b", "year"),
]


def _extract_context_window(text: str, start: int, end: int, window_tokens: int = 5) -> str:
    """Extract ±N tokens around a span."""
    # Simple tokenization by whitespace
    before = text[:start].split()
    after = text[end:].split()
    
    prefix = " ".join(before[-window_tokens:]) if before else ""
    suffix = " ".join(after[:window_tokens]) if after else ""
    span = text[start:end]
    
    parts = [p for p in [prefix, span, suffix] if p]
    return " ".join(parts)


def extract_time_anchors(text: str) -> list[Anchor]:
    """
    Extract time anchors from text using date patterns.
    
    Detects: YYYY-MM-DD, YYYY-MM, YYYY, Q1 2024, January 2024, etc.
    """
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0
    
    for pattern, pattern_type in TIME_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            
            # Skip if overlapping with already extracted anchor
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            
            counter += 1
            anchors.append(Anchor(
                anchor_id=f"t{counter}",
                kind=AnchorKind.TIME,
                span_text=match.group(0),
                char_start=start,
                char_end=end,
                context_window=_extract_context_window(text, start, end),
            ))
            seen_spans.add((start, end))
    
    return anchors


# ================= Numeric Anchors =================

# Numeric patterns with units/context
NUMERIC_PATTERNS = [
    # Currency: $1,234.56, €100, ¥50000
    (r"[$€£¥]\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|m|b))?\b", "currency"),
    # Percentage: 5%, 12.5%
    (r"\b\d+(?:\.\d+)?\s*%", "percentage"),
    # Large numbers with separators: 1,234,567
    (r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", "large_number"),
    # Numbers with units: 100km, 50kg, 25°C
    (r"\b\d+(?:\.\d+)?\s*(?:km|m|cm|mm|kg|g|mg|lb|oz|°[CF]|mph|kph)\b", "number_with_unit"),
    # Decimal numbers: 3.14, 0.5
    (r"\b\d+\.\d+\b", "decimal"),
    # Integer sequences of 2+ digits (avoid single digits)
    (r"\b\d{2,}\b", "integer"),
]


def extract_numeric_anchors(text: str) -> list[Anchor]:
    """
    Extract numeric anchors from text.
    
    Captures digit sequences with ±5 token context window.
    Does NOT infer meaning - purely structural extraction.
    """
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0
    
    for pattern, pattern_type in NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            
            # Skip if overlapping with already extracted anchor
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            
            counter += 1
            anchors.append(Anchor(
                anchor_id=f"n{counter}",
                kind=AnchorKind.NUMBER,
                span_text=match.group(0),
                char_start=start,
                char_end=end,
                context_window=_extract_context_window(text, start, end),
            ))
            seen_spans.add((start, end))
    
    return anchors


# ================= Quote Anchors =================

# Quote patterns for various quotation marks
QUOTE_PATTERNS = [
    # English double quotes
    (r'"([^"]+)"', "double_quote"),
    # English single quotes (careful: apostrophes can trigger)
    (r"'([^']+)'", "single_quote"),
    # Curly double quotes (Unicode)
    (r'[\u201c]([^\u201d]+)[\u201d]', "curly_double"),
    # Curly single quotes (Unicode)
    (r'[\u2018]([^\u2019]+)[\u2019]', "curly_single"),
    # French guillemets
    (r'«([^»]+)»', "guillemet"),
    # German quotes
    (r'„([^"]+)"', "german_quote"),
]


def extract_quote_anchors(text: str) -> list[Anchor]:
    """
    Extract quoted spans from text.
    
    Detects: "...", '...', "...", «...», etc.
    Extracts exact span only.
    """
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0
    
    for pattern, pattern_type in QUOTE_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            
            # Skip very short quotes (likely punctuation artifacts)
            if end - start < 5:
                continue
            
            # Skip if overlapping
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            
            counter += 1
            anchors.append(Anchor(
                anchor_id=f"q{counter}",
                kind=AnchorKind.QUOTE,
                span_text=match.group(0),
                char_start=start,
                char_end=end,
                context_window=_extract_context_window(text, start, end),
            ))
            seen_spans.add((start, end))
    
    return anchors


# ================= Combined Extraction =================

def extract_all_anchors(text: str) -> list[Anchor]:
    """
    Extract all deterministic anchors from text.
    
    Combines time, numeric, and quote anchors.
    Returns sorted by char_start position.
    
    Emits trace event with anchor counts.
    """
    if not text:
        return []
    
    time_anchors = extract_time_anchors(text)
    numeric_anchors = extract_numeric_anchors(text)
    quote_anchors = extract_quote_anchors(text)
    
    all_anchors = time_anchors + numeric_anchors + quote_anchors
    all_anchors.sort(key=lambda a: a.char_start)
    
    # Emit trace event
    Trace.event("claims.coverage.anchors", {
        "counts": {
            "time": len(time_anchors),
            "number": len(numeric_anchors),
            "quote": len(quote_anchors),
            "total": len(all_anchors),
        },
        "anchors": [
            {
                "anchor_id": a.anchor_id,
                "kind": a.kind.value,
                "preview": a.span_text[:30],
            }
            for a in all_anchors[:20]  # Limit trace size
        ],
    })
    
    return all_anchors


def get_anchor_ids(anchors: list[Anchor]) -> set[str]:
    """Get set of all anchor IDs."""
    return {a.anchor_id for a in anchors}


def anchors_to_prompt_context(anchors: list[Anchor]) -> str:
    """
    Format anchors as context for LLM prompts.
    
    Used in gap-fill repair to show which anchors need coverage.
    """
    if not anchors:
        return ""
    
    lines = []
    for a in anchors:
        lines.append(f"- [{a.anchor_id}] {a.kind.value}: \"{a.span_text}\" (context: {a.context_window[:60]}...)")
    
    return "\n".join(lines)
