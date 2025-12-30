# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Verdict Data Contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel

from spectrue_core.schema.policy import (
    DecisionPath,
    ErrorState,
    VerdictPolicy,
    DEFAULT_POLICY,
)
from spectrue_core.schema.signals import EvidenceSignals, LocaleDecision, TimeWindow


class VerdictStatus(str, Enum):
    """Claim assessment outcome - DERIVED from scores + policy."""
    
    VERIFIED = "verified"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class VerdictState(str, Enum):
    """Tier-dominant verdict state independent from score."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICTED = "conflicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VerdictHighlight(SchemaModel):
    """Minimal structured explanation for one assertion."""
    
    assertion_id: str
    stance: Literal["SUPPORTS", "REFUTES", "MIXED", "NO_EVIDENCE"]
    top_evidence_ids: list[str] = Field(default_factory=list)
    note: str = Field(default="")
    

class Verdict(SchemaModel):
    """
    Complete verdict output - the Data Contract.
    
    Key: confidence_score = 0.0 by default (Blind until proven seeing).
    """
    
    # Core Scores
    veracity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # Critical Fix: Default confidence is 0.0 (Blind until proven seeing)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    verdict_state: VerdictState | None = None
    
    error_state: ErrorState = Field(default=ErrorState.OK)
    decision_path: DecisionPath = Field(default=DecisionPath.WEB)
    signals: EvidenceSignals = Field(default_factory=EvidenceSignals)

    time_window: TimeWindow | None = None
    locale_decision: LocaleDecision | None = None
    
    summary: str = Field(default="")
    rationale: str = Field(default="")
    highlights: list[VerdictHighlight] = Field(default_factory=list)
    
    # Backward compat (optional RGBA)
    danger_score: float | None = None
    style_score: float | None = None
    explainability_score: float | None = None
    
    def status(self, policy: VerdictPolicy | None = None) -> VerdictStatus:
        """
        Derive status from scores + policy.
        
        Safety Rule: If no quotes and confidence > max_confidence_without_quotes,
        force AMBIGUOUS (side-effect free - doesn't mutate scores).
        """
        if policy is None:
            policy = DEFAULT_POLICY
            
        # Check for pipeline errors first
        if self.error_state in policy.unknown_if_error_states:
            return VerdictStatus.UNKNOWN
            
        # Safety Check: If confident but no quotes (and policy forbids it), force Ambiguous
        if (
            policy.max_confidence_without_quotes is not None 
            and self.signals.coverage.assertions_with_quotes == 0
            and self.confidence_score > policy.max_confidence_without_quotes
        ):
            return VerdictStatus.AMBIGUOUS

        # Verified path
        if self.veracity_score >= policy.verified_veracity_threshold:
            if self.confidence_score >= policy.min_confidence_for_verified:
                return VerdictStatus.VERIFIED
            return VerdictStatus.AMBIGUOUS
            
        # Refuted path
        if self.veracity_score <= policy.refuted_veracity_threshold:
            if self.confidence_score >= policy.min_confidence_for_refuted:
                return VerdictStatus.REFUTED
            return VerdictStatus.AMBIGUOUS
            
        return VerdictStatus.AMBIGUOUS
    
    @property
    def status_default(self) -> VerdictStatus:
        """Shortcut for status(DEFAULT_POLICY)."""
        return self.status(DEFAULT_POLICY)
    
    def is_error(self) -> bool:
        """Check if pipeline encountered an error."""
        return self.error_state != ErrorState.OK
    
    def has_evidence(self) -> bool:
        """Check if any evidence was retrieved."""
        return self.signals.has_readable_sources
    
    def to_summary_dict(self) -> dict[str, Any]:
        """Export as summary dict with derived status."""
        return {
            "status": self.status_default.value,
            "veracity_score": self.veracity_score,
            "confidence_score": self.confidence_score,
            "error_state": self.error_state.value,
            "decision_path": self.decision_path.value,
            "summary": self.summary,
            "has_evidence": self.has_evidence(),
        }
