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
"""
Embedding Utilities for ClaimGraph B-Stage (Shim)

Redirects to spectrue_core.adapters.embedding_client.
"""

from spectrue_core.adapters.embedding_client import (
    EmbeddingClient,
    cosine_similarity,
    _text_hash,
)

__all__ = [
    "EmbeddingClient",
    "cosine_similarity",
    "_text_hash",
]
