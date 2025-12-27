# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""M109: Embeddings module."""

from spectrue_core.embeddings.embed_service import (
    EmbedService,
    extract_best_quote,
    split_sentences,
)

__all__ = ["EmbedService", "extract_best_quote", "split_sentences"]
