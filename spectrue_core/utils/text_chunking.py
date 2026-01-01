# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Coverage Sampler & Text Chunking Utilities.

Provides structure-aware text splitting to prevent blind truncation
during article cleaning logic.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TextChunk:
    """A chunk of text from the original article."""
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    section_path: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.text)


class CoverageSampler:
    """
    Splits long text into manageable chunks while preserving structural boundaries.
    """

    def __init__(self):
        # Paragraph splitter: double newline or common block starts
        self.split_pattern = re.compile(r"\n\s*\n")
        # Heading detector (markdown style)
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, text: str, max_chunk_chars: int = 6000) -> list[TextChunk]:
        """
        Split text into chunks not exceeding max_chunk_chars.
        
        Preserves paragraph integrity where possible.
        Tracking offsets relative to original text.
        """
        if not text:
            return []

        if len(text) <= max_chunk_chars:
            return [TextChunk(
                chunk_id=str(uuid.uuid4())[:8],
                text=text,
                char_start=0,
                char_end=len(text),
                section_path=["root"]
            )]

        chunks: list[TextChunk] = []
        current_text = ""
        current_start = 0
        current_section: list[str] = ["root"]

        # Iterate over paragraphs (blocks)
        # We define a block as text ending with \n\n or end of string
        cursor = 0

        # Generator yielding (block_text, start_offset, end_offset)
        for block, start, end in self._iter_blocks(text):
            # Check for heading
            # Ideally we'd update current_section here, but simple logic for now
            # is just to append block.

            # If adding this block exceeds max, flush current
            if len(current_text) + len(block) > max_chunk_chars:
                if current_text:
                    chunks.append(TextChunk(
                        chunk_id=str(uuid.uuid4())[:8],
                        text=current_text,
                        char_start=current_start,
                        char_end=cursor, # Roughly end of last added block
                        section_path=list(current_section)
                    ))
                    # Reset
                    current_text = ""
                    current_start = start # Start of new chunk is start of this block

                # If single block is huge, we must force split it
                if len(block) > max_chunk_chars:
                    # Force split huge block
                    chunks.extend(self._force_split(block, start, max_chunk_chars))
                    cursor = end
                    continue

            # Append block
            if not current_text:
                current_start = start

            current_text += block
            cursor = end

        # Flush remainder
        if current_text:
            chunks.append(TextChunk(
                chunk_id=str(uuid.uuid4())[:8],
                text=current_text,
                char_start=current_start,
                char_end=len(text), # Up to the very end
                section_path=list(current_section)
            ))

        # Correction: ensure last chunk covers up to len(text) if gaps
        # But our iteration covers all chars if `_iter_blocks` is correct

        return chunks

    def merge(self, cleaned_chunks: list[str]) -> str:
        """
        Merge cleaned chunks into a single text with visual separators.
        """
        separator = "\n\n--- [SECTION BREAK] ---\n\n"
        return separator.join(c.strip() for c in cleaned_chunks if c.strip())

    def _iter_blocks(self, text: str) -> Iterator[tuple[str, int, int]]:
        """Yields (text, start, end) for each paragraph/block."""
        prev_end = 0
        for match in self.split_pattern.finditer(text):
            _, end = match.span()
            # Include the separator in the block
            full_block_text = text[prev_end:end]

            if full_block_text:
                yield full_block_text, prev_end, end

            prev_end = end

        # Last segment
        if prev_end < len(text):
            yield text[prev_end:], prev_end, len(text)

    def _force_split(self, text: str, global_offset: int, max_chars: int) -> list[TextChunk]:
        """Naively split a huge block by character limit."""
        chunks = []
        for i in range(0, len(text), max_chars):
            end = min(i + max_chars, len(text))
            chunk_text = text[i:end]
            chunks.append(TextChunk(
                chunk_id=str(uuid.uuid4())[:8],
                text=chunk_text,
                char_start=global_offset + i,
                char_end=global_offset + end,
                section_path=["root", "oversized"]
            ))
        return chunks
