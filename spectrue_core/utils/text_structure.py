# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Text structure extraction utilities.

Provides sentence and paragraph segmentation for context excerpt building
in per-claim deep analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    """A segment of text with position information."""
    text: str
    start: int  # char index in document
    end: int    # char index in document
    index: int  # segment index (0-based)


@dataclass(frozen=True)
class TextStructure:
    """
    Parsed structure of a document.
    
    Contains sentences and paragraphs with their positions.
    """
    sentences: tuple[TextSegment, ...]
    paragraphs: tuple[TextSegment, ...]
    
    def find_sentence_at(self, char_pos: int) -> TextSegment | None:
        """Find the sentence containing the given character position."""
        for seg in self.sentences:
            if seg.start <= char_pos < seg.end:
                return seg
        return None
    
    def find_paragraph_at(self, char_pos: int) -> TextSegment | None:
        """Find the paragraph containing the given character position."""
        for seg in self.paragraphs:
            if seg.start <= char_pos < seg.end:
                return seg
        return None
    
    def get_sentence_window(
        self, 
        sentence_index: int, 
        window_size: int = 1
    ) -> tuple[TextSegment, ...]:
        """
        Get a window of sentences around the given sentence index.
        
        Args:
            sentence_index: Central sentence index
            window_size: Number of sentences on each side (default 1 = claim + 2 neighbors)
        
        Returns:
            Tuple of sentences in the window
        """
        if not self.sentences:
            return ()
        
        start_idx = max(0, sentence_index - window_size)
        end_idx = min(len(self.sentences), sentence_index + window_size + 1)
        return self.sentences[start_idx:end_idx]


# Sentence boundary patterns
# Handles common sentence terminators with proper space handling
_SENTENCE_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ0-9"])|'  # After .!? followed by uppercase or digit
    r'(?<=[.!?])\s*\n',  # After .!? at end of line
    re.UNICODE
)

# Alternative pattern for simpler sentence splitting
_SIMPLE_SENTENCE_PATTERN = re.compile(
    r'(?<=[.!?])\s+',
    re.UNICODE
)

# Paragraph boundary patterns
_PARAGRAPH_PATTERN = re.compile(
    r'\n\s*\n|\r\n\s*\r\n',  # Double newline (with optional whitespace)
    re.UNICODE
)


def extract_sentences(text: str) -> tuple[TextSegment, ...]:
    """
    Extract sentences from text with position information.
    
    Uses regex-based sentence boundary detection. Falls back to
    simpler splitting if the main pattern produces no splits.
    
    Args:
        text: Input text
    
    Returns:
        Tuple of TextSegment objects for each sentence
    """
    if not text or not text.strip():
        return ()
    
    # Try main pattern first
    parts = _SENTENCE_PATTERN.split(text)
    
    # If no splits, try simpler pattern
    if len(parts) <= 1:
        parts = _SIMPLE_SENTENCE_PATTERN.split(text)
    
    # If still no splits, return whole text as one sentence
    if len(parts) <= 1:
        stripped = text.strip()
        if stripped:
            return (TextSegment(text=stripped, start=0, end=len(text), index=0),)
        return ()
    
    segments: list[TextSegment] = []
    current_pos = 0
    
    for idx, part in enumerate(parts):
        if not part:
            continue
        
        # Find actual position in original text
        start = text.find(part, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(part)
        
        stripped = part.strip()
        if stripped:
            segments.append(TextSegment(
                text=stripped,
                start=start,
                end=end,
                index=len(segments),
            ))
        
        current_pos = end
    
    return tuple(segments)


def extract_paragraphs(text: str) -> tuple[TextSegment, ...]:
    """
    Extract paragraphs from text with position information.
    
    Paragraphs are separated by double newlines.
    
    Args:
        text: Input text
    
    Returns:
        Tuple of TextSegment objects for each paragraph
    """
    if not text or not text.strip():
        return ()
    
    parts = _PARAGRAPH_PATTERN.split(text)
    
    # If no splits, return whole text as one paragraph
    if len(parts) <= 1:
        stripped = text.strip()
        if stripped:
            return (TextSegment(text=stripped, start=0, end=len(text), index=0),)
        return ()
    
    segments: list[TextSegment] = []
    current_pos = 0
    
    for part in parts:
        if not part:
            continue
        
        # Find actual position in original text
        start = text.find(part, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(part)
        
        stripped = part.strip()
        if stripped:
            segments.append(TextSegment(
                text=stripped,
                start=start,
                end=end,
                index=len(segments),
            ))
        
        current_pos = end
    
    return tuple(segments)


def extract_text_structure(text: str) -> TextStructure:
    """
    Extract full text structure including sentences and paragraphs.
    
    Args:
        text: Input text
    
    Returns:
        TextStructure with sentences and paragraphs
    """
    return TextStructure(
        sentences=extract_sentences(text),
        paragraphs=extract_paragraphs(text),
    )


def find_claim_position(
    text: str,
    claim_text: str,
    structure: TextStructure | None = None,
) -> tuple[int, int, int | None, int | None]:
    """
    Find the position of a claim in the document.
    
    Returns:
        Tuple of (span_start, span_end, sentence_index, paragraph_index)
        sentence_index and paragraph_index may be None if not found
    """
    if structure is None:
        structure = extract_text_structure(text)
    
    # Try exact match first
    span_start = text.find(claim_text)
    
    if span_start == -1:
        # Try normalized match (whitespace collapsed)
        normalized_text = ' '.join(text.split())
        normalized_claim = ' '.join(claim_text.split())
        norm_pos = normalized_text.find(normalized_claim)
        
        if norm_pos == -1:
            # Could not find claim, return beginning of document
            return 0, min(len(claim_text), len(text)), None, None
        
        # Approximate position in original text
        span_start = norm_pos
    
    span_end = span_start + len(claim_text)
    
    # Find sentence and paragraph indices
    sentence_index: int | None = None
    paragraph_index: int | None = None
    
    sentence = structure.find_sentence_at(span_start)
    if sentence is not None:
        sentence_index = sentence.index
    
    paragraph = structure.find_paragraph_at(span_start)
    if paragraph is not None:
        paragraph_index = paragraph.index
    
    return span_start, span_end, sentence_index, paragraph_index
