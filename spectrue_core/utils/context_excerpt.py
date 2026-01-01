# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Context excerpt extraction for deep analysis.

Provides sentence-window and paragraph-level context excerpt selection
for per-claim ClaimFrame building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.utils.text_structure import TextStructure

from spectrue_core.utils.text_structure import (
    TextSegment,
    extract_text_structure,
    find_claim_position,
)


@dataclass(frozen=True)
class ExcerptResult:
    """Result of context excerpt extraction."""
    text: str
    span_start: int
    span_end: int
    sentence_index: int | None
    paragraph_index: int | None
    sentence_window: tuple[int, int] | None  # (start_idx, end_idx)
    method: str  # "sentence_window", "paragraph", "fallback"


def build_sentence_window_excerpt(
    document_text: str,
    claim_text: str,
    window_size: int = 1,
    max_chars: int = 500,
) -> ExcerptResult | None:
    """
    Build a context excerpt using sentence-window selection.
    
    Extracts the claim sentence plus neighboring sentences.
    
    Args:
        document_text: Full document text
        claim_text: The claim to find context for
        window_size: Number of sentences on each side (default 1)
        max_chars: Maximum excerpt length in characters
    
    Returns:
        ExcerptResult with sentence-based excerpt, or None if not found
    """
    if not document_text or not claim_text:
        return None

    structure = extract_text_structure(document_text)
    span_start, span_end, sentence_index, paragraph_index = find_claim_position(
        document_text, claim_text, structure
    )

    if sentence_index is None:
        return None

    # Get sentence window
    window = structure.get_sentence_window(sentence_index, window_size)
    if not window:
        return None

    # Build excerpt text from window
    excerpt_parts = [seg.text for seg in window]
    excerpt_text = " ".join(excerpt_parts)

    # Truncate if too long
    if len(excerpt_text) > max_chars:
        # Prioritize sentences closest to claim
        center_idx = sentence_index
        start_offset = sentence_index - window[0].index

        # Keep claim sentence always
        result_parts = [window[start_offset].text]
        remaining = max_chars - len(result_parts[0])

        # Add neighbors alternating left/right
        left_idx = start_offset - 1
        right_idx = start_offset + 1

        while remaining > 50 and (left_idx >= 0 or right_idx < len(window)):
            if right_idx < len(window):
                part = window[right_idx].text
                if len(part) < remaining:
                    result_parts.append(part)
                    remaining -= len(part) + 1
                right_idx += 1

            if left_idx >= 0:
                part = window[left_idx].text
                if len(part) < remaining:
                    result_parts.insert(0, part)
                    remaining -= len(part) + 1
                left_idx -= 1

        excerpt_text = " ".join(result_parts)

    # Calculate actual span in document
    excerpt_start = window[0].start
    excerpt_end = window[-1].end

    window_range = (window[0].index, window[-1].index)

    return ExcerptResult(
        text=excerpt_text,
        span_start=excerpt_start,
        span_end=excerpt_end,
        sentence_index=sentence_index,
        paragraph_index=paragraph_index,
        sentence_window=window_range,
        method="sentence_window",
    )


def build_paragraph_excerpt(
    document_text: str,
    claim_text: str,
    max_chars: int = 600,
) -> ExcerptResult | None:
    """
    Build a context excerpt using paragraph-level selection.
    
    Fallback when sentence-window doesn't work (e.g., claim spans paragraphs).
    
    Args:
        document_text: Full document text
        claim_text: The claim to find context for
        max_chars: Maximum excerpt length
    
    Returns:
        ExcerptResult with paragraph-based excerpt, or None if not found
    """
    if not document_text or not claim_text:
        return None

    structure = extract_text_structure(document_text)
    span_start, span_end, sentence_index, paragraph_index = find_claim_position(
        document_text, claim_text, structure
    )

    if paragraph_index is None:
        return None

    if paragraph_index >= len(structure.paragraphs):
        return None

    paragraph = structure.paragraphs[paragraph_index]
    excerpt_text = paragraph.text

    # Truncate from both ends if needed
    if len(excerpt_text) > max_chars:
        # Find claim position within paragraph
        claim_pos = excerpt_text.find(claim_text)
        if claim_pos == -1:
            claim_pos = len(excerpt_text) // 2

        # Center the window around claim
        half_window = max_chars // 2
        start = max(0, claim_pos - half_window)
        end = min(len(excerpt_text), claim_pos + len(claim_text) + half_window)

        # Adjust to not cut words
        if start > 0:
            space_pos = excerpt_text.find(" ", start)
            if space_pos != -1 and space_pos < start + 20:
                start = space_pos + 1

        if end < len(excerpt_text):
            space_pos = excerpt_text.rfind(" ", 0, end)
            if space_pos != -1 and space_pos > end - 20:
                end = space_pos

        excerpt_text = excerpt_text[start:end].strip()
        if start > 0:
            excerpt_text = "..." + excerpt_text
        if end < len(paragraph.text):
            excerpt_text = excerpt_text + "..."

    return ExcerptResult(
        text=excerpt_text,
        span_start=paragraph.start,
        span_end=paragraph.end,
        sentence_index=sentence_index,
        paragraph_index=paragraph_index,
        sentence_window=None,
        method="paragraph",
    )


def build_fallback_excerpt(
    document_text: str,
    claim_text: str,
    context_chars: int = 200,
) -> ExcerptResult:
    """
    Build a fallback context excerpt when structured methods fail.
    
    Simply takes characters around the claim position.
    
    Args:
        document_text: Full document text
        claim_text: The claim to find context for
        context_chars: Number of context characters on each side
    
    Returns:
        ExcerptResult with character-based excerpt
    """
    if not document_text:
        return ExcerptResult(
            text=claim_text[:500] if claim_text else "",
            span_start=0,
            span_end=len(claim_text) if claim_text else 0,
            sentence_index=None,
            paragraph_index=None,
            sentence_window=None,
            method="fallback",
        )

    # Find claim position
    span_start = document_text.find(claim_text)
    if span_start == -1:
        span_start = 0
        span_end = min(len(claim_text), len(document_text))
    else:
        span_end = span_start + len(claim_text)

    # Extract with context
    excerpt_start = max(0, span_start - context_chars)
    excerpt_end = min(len(document_text), span_end + context_chars)

    excerpt_text = document_text[excerpt_start:excerpt_end].strip()

    # Add ellipsis if truncated
    if excerpt_start > 0:
        # Find first space to not cut words
        first_space = excerpt_text.find(" ")
        if first_space != -1 and first_space < 20:
            excerpt_text = excerpt_text[first_space + 1:]
        excerpt_text = "..." + excerpt_text

    if excerpt_end < len(document_text):
        last_space = excerpt_text.rfind(" ")
        if last_space != -1 and last_space > len(excerpt_text) - 20:
            excerpt_text = excerpt_text[:last_space]
        excerpt_text = excerpt_text + "..."

    return ExcerptResult(
        text=excerpt_text,
        span_start=excerpt_start,
        span_end=excerpt_end,
        sentence_index=None,
        paragraph_index=None,
        sentence_window=None,
        method="fallback",
    )


def build_context_excerpt(
    document_text: str,
    claim_text: str,
    window_size: int = 1,
    max_chars: int = 500,
) -> ExcerptResult:
    """
    Build the best available context excerpt for a claim.
    
    Tries methods in order:
    1. Sentence-window (claim sentence + neighbors)
    2. Paragraph-level (claim's paragraph)
    3. Fallback (character-based around claim)
    
    Args:
        document_text: Full document text
        claim_text: The claim to find context for
        window_size: Sentence window size (default 1 = +/- 1)
        max_chars: Maximum excerpt length
    
    Returns:
        ExcerptResult with the best available excerpt
    """
    # Try sentence-window first (preferred)
    result = build_sentence_window_excerpt(
        document_text, claim_text, window_size, max_chars
    )
    if result is not None:
        return result

    # Try paragraph-level
    result = build_paragraph_excerpt(document_text, claim_text, max_chars)
    if result is not None:
        return result

    # Fallback to character-based
    return build_fallback_excerpt(document_text, claim_text)