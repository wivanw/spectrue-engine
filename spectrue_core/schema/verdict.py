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
"""Structured Verdict Pydantic Models (Schema Adapter)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel
from spectrue_core.domain.verification.verdict.model import (
    AnalysisMode,
    ScoringMode,
    VerdictStatus,
    VerdictState,
    AssertionVerdict as DomainAssertionVerdict,
    ClaimVerdict as DomainClaimVerdict,
    StructuredDebug as DomainStructuredDebug,
    StructuredVerdict as DomainStructuredVerdict,
    BeliefState,
    EvidenceFlowInput,
    EvidenceCollection,
    ProgressCallback,
)

__all__ = [
    "AnalysisMode",
    "ScoringMode",
    "VerdictStatus",
    "VerdictState",
    "AssertionVerdict",
    "ClaimVerdict",
    "StructuredDebug",
    "StructuredVerdict",
    "BeliefState",
    "EvidenceFlowInput",
    "EvidenceCollection",
    "ProgressCallback",
]


class AssertionVerdict(SchemaModel, DomainAssertionVerdict):
    """Verdict for a single assertion."""
    assertion_key: str
    dimension: str = "FACT"
    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_count: int = 0
    supporting_urls: list[str] = Field(default_factory=list)
    rationale: str = ""


class ClaimVerdict(SchemaModel, DomainClaimVerdict):
    """Verdict for a claim (aggregated from assertion verdicts)."""
    claim_id: str
    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    verdict: VerdictStatus = VerdictStatus.AMBIGUOUS
    verdict_state: VerdictState = VerdictState.INSUFFICIENT_EVIDENCE
    verdict_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: str = "low"
    reasons_short: list[str] = Field(default_factory=list)
    reasons_expert: dict[str, Any] = Field(default_factory=dict)
    assertion_verdicts: list[AssertionVerdict] = Field(default_factory=list)
    evidence_count: int = 0
    fact_assertions_verified: int = 0
    fact_assertions_total: int = 0
    reason: str = ""
    key_evidence: list[str] = Field(default_factory=list)
    prior_score: float = Field(default=-1.0)
    prior_reason: str = ""


class StructuredDebug(SchemaModel, DomainStructuredDebug):
    """Debug information (not exposed to users)."""
    per_claim: dict[str, Any] = Field(default_factory=dict)
    dropped_evidence: list[dict[str, Any]] = Field(default_factory=list)
    content_unavailable_count: int = 0
    processing_notes: list[str] = Field(default_factory=list)


class StructuredVerdict(SchemaModel, DomainStructuredVerdict):
    """Complete verdict output from scoring."""
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    verified_score: float = Field(default=-1.0)
    explainability_score: float = Field(default=-1.0)
    danger_score: float = Field(default=-1.0)
    style_score: float = Field(default=-1.0)
    rationale: str = ""
    structured_debug: StructuredDebug | None = None
    overall_confidence: float = Field(default=-1.0)
    evidence_gaps: list[str] = Field(default_factory=list)
