# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Retrieval quality evaluation helpers."""

from __future__ import annotations

from spectrue_core.verification.evidence_pack import score_evidence_likeness
from spectrue_core.verification.source_utils import score_source_quality


def evaluate_retrieval_confidence(sources: list[dict]) -> dict[str, float]:
    """
    Compute retrieval confidence from relevance, evidence-likeness, and source quality.
    """
    relevance_scores = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        score = src.get("relevance_score")
        if score is None:
            continue
        try:
            relevance_scores.append(float(score))
        except Exception:
            continue

    relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    evidence_likeness = score_evidence_likeness(sources)
    source_quality = score_source_quality(sources)
    retrieval_confidence = (
        (0.5 * relevance)
        + (0.3 * evidence_likeness)
        + (0.2 * source_quality)
    )
    return {
        "relevance_score": relevance,
        "evidence_likeness_score": evidence_likeness,
        "source_quality_score": source_quality,
        "retrieval_confidence": retrieval_confidence,
    }
