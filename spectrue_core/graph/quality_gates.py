# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
ClaimGraph Quality Gates

Isolates gate logic from `ClaimGraphBuilder` so the main builder reads as a
sequence of steps.
"""

from __future__ import annotations

import logging

from spectrue_core.graph.types import CandidateEdge, TypedEdge, GraphResult

logger = logging.getLogger(__name__)


def check_kept_ratio_within_topic(
    *,
    min_kept_ratio: float,
    max_kept_ratio: float,
    candidates: list[CandidateEdge],
    kept_edges: list[TypedEdge],
    result: GraphResult,
) -> bool:
    """
    Check if kept_ratio is within acceptable bounds.

    M75: Exclude cross-topic edges from the ratio calculation.
    M76: If ratio is low, treat as sparse-but-valid (do not disable).
    """
    cross_topic_pairs = {(c.src_id, c.dst_id) for c in candidates if c.cross_topic}

    kept_within = [e for e in kept_edges if (e.src_id, e.dst_id) not in cross_topic_pairs]
    cand_within = [c for c in candidates if not c.cross_topic]

    result.within_topic_edges_count = len(cand_within)
    result.cross_topic_edges_count = len(candidates) - len(cand_within)

    if not cand_within:
        numerator = len(kept_edges)
        denominator = len(candidates)
        logger.debug("[M75] No within-topic candidates, using overall ratio")
    else:
        numerator = len(kept_within)
        denominator = len(cand_within)

    kept_ratio = numerator / denominator if denominator > 0 else 0.0
    result.kept_ratio_within_topic = kept_ratio

    if kept_ratio < min_kept_ratio:
        logger.warning(
            "[M72] Quality gate: kept_ratio %.3f < min %.3f (within-focus)",
            kept_ratio,
            min_kept_ratio,
        )
        result.sparse_graph = True
        return False

    if kept_ratio > max_kept_ratio:
        logger.warning(
            "[M72] Quality gate: kept_ratio %.3f > max %.3f (within-focus)",
            kept_ratio,
            max_kept_ratio,
        )
        return False

    return True
