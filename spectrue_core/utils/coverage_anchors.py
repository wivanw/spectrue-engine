"""Deterministic coverage anchors for query planning."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class AnchorKind(Enum):
    """Anchor type classification."""

    TIME = "time"
    NUMBER = "number"
    QUOTE = "quote"


@dataclass
class Anchor:
    """
    A deterministic anchor extracted from text.
    """

    anchor_id: str
    kind: AnchorKind
    span_text: str
    char_start: int
    char_end: int
    context_window: str


# ================= Time Anchors =================

TIME_PATTERNS = [
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "iso_date"),
    (r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", "eu_date"),
    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "us_date"),
    (r"\b(\d{4})-(\d{2})\b", "year_month"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", "month_year"),
    (r"\b[Qq]([1-4])\s*(\d{4})\b", "quarter"),
    (r"\b(19\d{2}|20\d{2})\b", "year"),
]


def _extract_context_window(text: str, start: int, end: int, window_tokens: int = 5) -> str:
    """Extract +/- N tokens around a span."""
    before = text[:start].split()
    after = text[end:].split()

    prefix = " ".join(before[-window_tokens:]) if before else ""
    suffix = " ".join(after[:window_tokens]) if after else ""
    span = text[start:end]

    parts = [p for p in [prefix, span, suffix] if p]
    return " ".join(parts)


def extract_time_anchors(text: str) -> list[Anchor]:
    """Extract time anchors from text using date patterns."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in TIME_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"t{counter}",
                    kind=AnchorKind.TIME,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Numeric Anchors =================

NUMERIC_PATTERNS = [
    (r"[$€£¥]\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|m|b))?\b", "currency"),
    (r"\b\d+(?:\.\d+)?\s*%", "percentage"),
    (r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", "large_number"),
    (r"\b\d+(?:\.\d+)?\s*(?:km|m|cm|mm|kg|g|mg|lb|oz|°[CF]|mph|kph)\b", "number_with_unit"),
    (r"\b\d+\.\d+\b", "decimal"),
    (r"\b\d{2,}\b", "integer"),
]


def extract_numeric_anchors(text: str) -> list[Anchor]:
    """Extract numeric anchors from text."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"n{counter}",
                    kind=AnchorKind.NUMBER,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Quote Anchors =================

QUOTE_PATTERNS = [
    (r'"([^"]+)"', "double_quote"),
    (r"'([^']+)'", "single_quote"),
    (r"[“]([^”]+)[”]", "curly_double"),
    (r"[‘]([^’]+)[’]", "curly_single"),
    (r"«([^»]+)»", "guillemet"),
    (r'„([^"]+)"', "german_quote"),
]


def extract_quote_anchors(text: str) -> list[Anchor]:
    """Extract quoted spans from text."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in QUOTE_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()

            if end - start < 5:
                continue

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"q{counter}",
                    kind=AnchorKind.QUOTE,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Combined Extraction =================


def extract_all_anchors(text: str) -> list[Anchor]:
    """Extract all deterministic anchors from text."""
    if not text:
        return []

    time_anchors = extract_time_anchors(text)
    numeric_anchors = extract_numeric_anchors(text)
    quote_anchors = extract_quote_anchors(text)

    all_anchors = time_anchors + numeric_anchors + quote_anchors
    all_anchors.sort(key=lambda a: a.char_start)

    return all_anchors


def get_anchor_ids(anchors: list[Anchor]) -> set[str]:
    return {a.anchor_id for a in anchors}


def anchors_to_prompt_context(anchors: list[Anchor]) -> str:
    if not anchors:
        return ""

    lines = []
    for a in anchors:
        lines.append(
            f"- [{a.anchor_id}] {a.kind.value}: \"{a.span_text}\" (context: {a.context_window[:60]}...)"
        )

    return "\n".join(lines)
