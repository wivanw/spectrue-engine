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
"""Fallback token estimation utilities for LLM billing."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    word_count = len(text.split())
    if word_count == 0:
        return 0
    return max(1, int(word_count * 1.3))


def estimate_tokens_from_chars(char_count: int) -> int:
    """Estimate token count when only the prompt length in characters is known.

    Used by cost-aware routing, which sizes a prompt before it is built and so
    has no text to feed :func:`estimate_tokens`. Uses the same ~4 chars/token
    heuristic as the embedding fallback in ``billing.metering``.
    """
    if char_count <= 0:
        return 0
    return max(1, int(char_count // 4))


def estimate_completion_usage(
    *, input_text: str, output_text: str, instructions: str | None = None
) -> dict[str, int]:
    combined_input = " ".join(part for part in [instructions or "", input_text] if part).strip()
    return {
        "input_tokens": estimate_tokens(combined_input),
        "output_tokens": estimate_tokens(output_text),
    }
