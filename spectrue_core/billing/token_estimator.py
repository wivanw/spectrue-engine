# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Fallback token estimation utilities for LLM billing."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    word_count = len(text.split())
    if word_count == 0:
        return 0
    return max(1, int(word_count * 1.3))


def estimate_completion_usage(
    *, input_text: str, output_text: str, instructions: str | None = None
) -> dict[str, int]:
    combined_input = " ".join(part for part in [instructions or "", input_text] if part).strip()
    return {
        "input_tokens": estimate_tokens(combined_input),
        "output_tokens": estimate_tokens(output_text),
    }
