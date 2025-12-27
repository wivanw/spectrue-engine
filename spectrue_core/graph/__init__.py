# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M72: Hybrid ClaimGraph (B + C) Package

Two-stage sparse graph for claim prioritization:
- B-stage: cheap candidate generation (embeddings + adjacency)
- C-stage: LLM edge typing (GPT-5 nano)

Evidence-first principle: This system NEVER produces truth.
It only identifies WHICH claims to prioritize for verification.
"""

from spectrue_core.graph.types import (
    ClaimNode,
    ClaimPreGraphMeta,
    ClaimPostGraphMeta,
    CandidateEdge,
    TypedEdge,
    RankedClaim,
    DedupeResult,
    GraphResult,
    EdgeRelation,
)
from spectrue_core.graph.claim_graph import ClaimGraphBuilder

__all__ = [
    "ClaimGraphBuilder",
    "ClaimNode",
    "ClaimPreGraphMeta",
    "ClaimPostGraphMeta",
    "CandidateEdge",
    "TypedEdge",
    "RankedClaim",
    "DedupeResult",
    "GraphResult",
    "EdgeRelation",
]
