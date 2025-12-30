# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Structured Verdict Pydantic Models

Verdict is the OUTPUT of scoring.
It contains per-assertion verdicts for explainability.

Key Design Principles:
1. Per-assertion verdicts (not just per-claim)
2. Clear separation: FACT drives verdict, CONTEXT is modifier
3. Sentinel -1.0 = explicit "LLM didn't provide this"
4. structured_debug for diagnostics (not exposed to users)
"""

from enum import Enum
from typing import Any

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel


class VerdictStatus(str, Enum):
    """Verdict outcome for an assertion or claim."""
    VERIFIED = "verified"
    """Evidence confirms the assertion."""

    REFUTED = "refuted"
    """Evidence contradicts the assertion."""

    AMBIGUOUS = "ambiguous"
    """Insufficient or conflicting evidence."""

    PARTIALLY_VERIFIED = "partially_verified"
    """Some aspects verified, others not."""

    UNVERIFIED = "unverified"
    """No evidence found."""

    SATIRICAL = "satirical"
    """Content identified as satire/parody. Not a factual claim."""


class VerdictState(str, Enum):
    """Tier-dominant verdict state independent from score."""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICTED = "conflicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"



class AssertionVerdict(SchemaModel):
    """
    Verdict for a single assertion.
    
    This is the atomic verdict unit.
    Claim verdict is aggregated from assertion verdicts.
    """

    assertion_key: str
    """Which assertion this verdict is for."""

    dimension: str = "FACT"
    """FACT, CONTEXT, or INTERPRETATION."""

    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    """verified, refuted, ambiguous, etc."""

    score: float = Field(default=0.5, ge=0.0, le=1.0)
    """Verification score. 0=refuted, 0.5=ambiguous, 1=verified."""

    evidence_count: int = 0
    """How many evidence items support this verdict."""

    supporting_urls: list[str] = Field(default_factory=list)
    """URLs of evidence that influenced this verdict."""

    rationale: str = ""
    """Explanation for this specific assertion verdict."""


class ClaimVerdict(SchemaModel):
    """
    Verdict for a claim (aggregated from assertion verdicts).
    
    FACT assertions drive the verdict.
    CONTEXT assertions are modifiers.
    """

    claim_id: str
    """Which claim this verdict is for."""

    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    """Overall claim verdict."""

    verdict: VerdictStatus = VerdictStatus.AMBIGUOUS
    """Alias for status to match external contract."""

    verdict_state: VerdictState = VerdictState.INSUFFICIENT_EVIDENCE
    """Tier-dominant verdict state derived in code."""

    verdict_score: float = Field(default=0.5, ge=0.0, le=1.0)
    """Aggregated verification score."""

    confidence: str = "low"
    """Confidence label: low, medium, high."""

    reasons_short: list[str] = Field(default_factory=list)
    """Short bullet reasons for the verdict."""

    reasons_expert: dict[str, Any] = Field(default_factory=dict)
    """Expert-level evidence references and penalties."""

    # Per-assertion breakdown
    assertion_verdicts: list[AssertionVerdict] = Field(default_factory=list)
    """Individual verdicts for each assertion."""

    # Evidence summary
    evidence_count: int = 0
    """Total evidence items for this claim."""

    fact_assertions_verified: int = 0
    """How many FACT assertions are verified."""

    fact_assertions_total: int = 0
    """Total FACT assertions."""

    # Human-readable
    reason: str = ""
    """Summary explanation."""

    key_evidence: list[str] = Field(default_factory=list)
    """Most important evidence URLs."""


class StructuredDebug(SchemaModel):
    """
    Debug information (not exposed to users).
    
    Contains diagnostics for troubleshooting.
    """

    per_claim: dict[str, Any] = Field(default_factory=dict)
    """Per-claim debug info: assertion summaries, edge cases."""

    dropped_evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Evidence that was dropped and why."""

    content_unavailable_count: int = 0
    """How many sources had CONTENT_UNAVAILABLE."""

    processing_notes: list[str] = Field(default_factory=list)
    """Notes from processing pipeline."""


class StructuredVerdict(SchemaModel):
    """
    Complete verdict output from scoring.
    
    This is the full output with:
    - Per-claim verdicts
    - Global RGBA scores
    - Debug information (internal)
    
    Sentinel values:
    - -1.0 means "LLM didn't provide this value"
    - This is NOT a score, it's a diagnostic signal
    """

    # Per-claim verdicts
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    """Verdicts for each claim."""

    # Global RGBA scores (can be -1.0 sentinel)
    verified_score: float = Field(default=-1.0)
    """
    G channel: overall truthfulness.
    -1.0 = sentinel (LLM didn't provide).
    0.0-1.0 = actual score.
    """

    explainability_score: float = Field(default=-1.0)
    """
    A channel: how well evidence explains the verdict.
    Lowered when CONTENT_UNAVAILABLE sources exist.
    """

    danger_score: float = Field(default=-1.0)
    """
    R channel: potential harm if acted upon.
    Higher for health/politics misinformation.
    """

    style_score: float = Field(default=-1.0)
    """
    B channel: writing neutrality.
    0 = heavily biased, 1 = neutral journalism.
    """

    # Human-readable output
    rationale: str = ""
    """Overall explanation in user's language."""

    # Diagnostics (internal only)
    structured_debug: StructuredDebug | None = None
    """Debug info. Not exposed to end users."""

    # Legacy compatibility
    overall_confidence: float = Field(default=-1.0)
    """Legacy: same as verified_score."""

    evidence_gaps: list[str] = Field(default_factory=list)
    """Legacy: list of missing evidence types."""

    def is_complete(self) -> bool:
        """Check if all required scores are present (not sentinel)."""
        return all([
            self.verified_score >= 0,
            self.explainability_score >= 0,
            self.danger_score >= 0,
            self.style_score >= 0,
        ])

    def get_fact_verification_ratio(self) -> tuple[int, int]:
        """Get (verified_count, total_count) for FACT assertions."""
        verified = sum(cv.fact_assertions_verified for cv in self.claim_verdicts)
        total = sum(cv.fact_assertions_total for cv in self.claim_verdicts)
        return verified, total
