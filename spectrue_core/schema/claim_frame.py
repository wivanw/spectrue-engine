"""
Per-Claim Judging schema models (Shim).

This module re-exports domain types.
"""

from __future__ import annotations

from spectrue_core.domain.claims.frame import (
    ClaimFrame,
    ClaimResult,
    ClaimResultDict,
    ConfirmationCounts,
    ContextExcerpt,
    ContextMeta,
    DeepAnalysisResult,
    EvidenceItemFrame,
    EvidenceReference,
    EvidenceStats,
    EvidenceStanceStats,
    EvidenceSummary,
    JudgeOutput,
    JudgeOutputDict,
    RetrievalHop,
    RetrievalTrace,
    RGBAScore,
    EvidenceCleanlinessRecord,
)

__all__ = [
    "ClaimFrame",
    "ClaimResult",
    "ClaimResultDict",
    "ConfirmationCounts",
    "ContextExcerpt",
    "ContextMeta",
    "DeepAnalysisResult",
    "EvidenceItemFrame",
    "EvidenceReference",
    "EvidenceStats",
    "EvidenceStanceStats",
    "EvidenceSummary",
    "JudgeOutput",
    "JudgeOutputDict",
    "RetrievalHop",
    "RetrievalTrace",
    "RGBAScore",
    "EvidenceCleanlinessRecord",
]
