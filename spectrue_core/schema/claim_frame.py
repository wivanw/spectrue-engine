# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Per-Claim Judging schema models.

This module defines ClaimFrame and related entities for deep analysis mode,
where each claim is evaluated independently with its own context, evidence,
and judge output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextExcerpt:
    """
    User-visible text snippet around a claim.
    
    Derived from sentence or paragraph boundaries in the source document.
    """
    text: str
    source_type: str = "user_text"  # user_text, cleaned_article, etc.
    span_start: int = 0  # char index in document
    span_end: int = 0    # char index in document
    
    def __post_init__(self) -> None:
        if self.span_end < self.span_start:
            object.__setattr__(self, "span_end", self.span_start + len(self.text))


@dataclass(frozen=True)
class ContextMeta:
    """
    Structural metadata about the claim's position in the document.
    """
    document_id: str
    paragraph_index: int | None = None
    sentence_index: int | None = None
    sentence_window: tuple[int, int] | None = None  # (start_index, end_index)


@dataclass(frozen=True)
class EvidenceItemFrame:
    """
    A single evidence record scoped to a specific claim.
    
    Note: This is distinct from spectrue_core.schema.evidence.EvidenceItem
    which is used for pipeline-internal processing. EvidenceItemFrame is
    the API-facing model for per-claim results.
    """
    evidence_id: str
    claim_id: str
    url: str
    title: str | None = None
    source_tier: str | None = None  # A, A', B, C, D, UNKNOWN
    source_type: str | None = None
    stance: str | None = None  # SUPPORT, REFUTE, CONTEXT, IRRELEVANT
    quote: str | None = None
    snippet: str | None = None
    relevance: float | None = None  # 0..1


@dataclass(frozen=True)
class EvidenceStats:
    """
    Aggregate counts summarizing evidence coverage for a claim.
    """
    total_sources: int = 0
    support_sources: int = 0
    refute_sources: int = 0
    context_sources: int = 0
    high_trust_sources: int = 0
    direct_quotes: int = 0
    conflicting_evidence: bool = False
    missing_sources: bool = True
    missing_direct_quotes: bool = True
    
    def __post_init__(self) -> None:
        # Derive missing_sources from total_sources if not explicitly set
        if self.total_sources > 0:
            object.__setattr__(self, "missing_sources", False)
        if self.direct_quotes > 0:
            object.__setattr__(self, "missing_direct_quotes", False)


@dataclass(frozen=True)
class RetrievalHop:
    """
    A single retrieval step in the search process.
    """
    hop_index: int
    query: str
    decision: str  # CONTINUE, STOP, etc.
    reason: str
    phase_id: str | None = None
    query_type: str | None = None
    results_count: int = 0
    retrieval_eval: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalTrace:
    """
    Metadata describing how evidence was searched and filtered for a claim.
    """
    phases_completed: tuple[str, ...] = ()
    hops: tuple[RetrievalHop, ...] = ()
    stop_reason: str | None = None
    sufficiency_reason: str | None = None


@dataclass(frozen=True)
class ClaimFrame:
    """
    Per-claim input bundle for deep analysis mode.
    
    Contains all context and evidence needed for the judge to evaluate
    a single claim independently.
    """
    claim_id: str
    claim_text: str
    claim_language: str  # ISO-639-1 (e.g., "en", "uk")
    context_excerpt: ContextExcerpt
    context_meta: ContextMeta
    evidence_items: tuple[EvidenceItemFrame, ...] = ()
    evidence_stats: EvidenceStats = field(default_factory=EvidenceStats)
    retrieval_trace: RetrievalTrace = field(default_factory=RetrievalTrace)


@dataclass(frozen=True)
class EvidenceReference:
    """
    Reference to an evidence item with reasoning.
    """
    evidence_id: str
    reason: str


@dataclass(frozen=True)
class EvidenceSummary:
    """
    Optional structured summary of evidence for a claim.
    Produced by evidence summarizer before judge invocation.
    """
    supporting_evidence: tuple[EvidenceReference, ...] = ()
    refuting_evidence: tuple[EvidenceReference, ...] = ()
    contextual_evidence: tuple[EvidenceReference, ...] = ()
    evidence_gaps: tuple[str, ...] = ()
    conflicts_present: bool = False


@dataclass(frozen=True)
class RGBAScore:
    """
    RGBA score components.
    """
    r: float  # Danger (0..1)
    g: float  # Veracity (0..1)
    b: float  # Honesty (0..1)
    a: float  # Explainability (0..1)
    
    def to_dict(self) -> dict[str, float]:
        return {"R": self.r, "G": self.g, "B": self.b, "A": self.a}
    
    def to_list(self) -> list[float]:
        return [self.r, self.g, self.b, self.a]


@dataclass(frozen=True)
class JudgeOutput:
    """
    Raw judge output for a single claim.
    
    This must be returned to the frontend unchanged, preserving all fields
    exactly as produced by the LLM judge.
    """
    claim_id: str
    rgba: RGBAScore
    confidence: float  # 0..1
    verdict: str  # Supported, Refuted, NEI, etc.
    explanation: str
    sources_used: tuple[str, ...] = ()  # subset of EvidenceItemFrame.url
    missing_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimResult:
    """
    Complete result for a single claim in deep analysis mode.
    
    Combines the claim frame (input) with judge output and optional summary.
    """
    claim_frame: ClaimFrame
    judge_output: JudgeOutput
    evidence_summary: EvidenceSummary | None = None
    error: dict[str, Any] | None = None  # claim-level failures


@dataclass(frozen=True)
class DeepAnalysisResult:
    """
    Response structure for deep analysis mode.
    
    Contains only per-claim results with no aggregate verdict.
    """
    analysis_mode: str = "deep"
    claim_results: tuple[ClaimResult, ...] = ()


# Type aliases for convenience
ClaimFrameDict = dict[str, Any]
JudgeOutputDict = dict[str, Any]
ClaimResultDict = dict[str, Any]
