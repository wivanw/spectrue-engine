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
ClaimGraph Type Definitions (Legacy Shim)

Re-exports from spectrue_core.domain.graph_types.
"""

from spectrue_core.domain.claims.graph.types import (
    ClaimNode,
    ClaimPreGraphMeta,
    ClaimPostGraphMeta,
    CandidateEdge,
    TypedEdge,
    RankedClaim,
    DedupeResult,
    GraphResult,
    EdgeRelation,
    RELATION_MULTIPLIERS,
    STRUCTURAL_RELATIONS,
)

__all__ = [
    "ClaimNode",
    "ClaimPreGraphMeta",
    "ClaimPostGraphMeta",
    "CandidateEdge",
    "TypedEdge",
    "RankedClaim",
    "DedupeResult",
    "GraphResult",
    "EdgeRelation",
    "RELATION_MULTIPLIERS",
    "STRUCTURAL_RELATIONS",
]
