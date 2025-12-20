# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M70: EvidenceItem Pydantic Model

Evidence is a DOWNSTREAM CONSUMER of the schema.
It maps to specific assertion_key, not just claim_id.

Key Design Principles:
1. Evidence links to assertion_key (not just raw claim text)
2. CONTENT_UNAVAILABLE status is explicit (not silent drop)
3. Stance is per-assertion, enabling granular verification
"""

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceStance(str, Enum):
    """
    Position of evidence relative to an assertion.
    
    LLM determines stance. Code uses it for aggregation.
    """
    SUPPORT = "SUPPORT"
    """Evidence confirms the assertion."""

    REFUTE = "REFUTE"
    """Evidence contradicts the assertion."""

    MIXED = "MIXED"
    """Evidence partially supports, partially contradicts."""

    MENTION = "MENTION"
    """Evidence mentions the topic but doesn't take a stance."""

    IRRELEVANT = "IRRELEVANT"
    """Evidence is not related to this assertion."""


class ContentStatus(str, Enum):
    """
    Status of content retrieval.
    
    Critical for handling treasury.gov/OFAC empty snippets:
    - Keep the evidence item
    - Lower explainability_score
    - Claim stays AMBIGUOUS, not REFUTED
    """
    AVAILABLE = "AVAILABLE"
    """Content was successfully retrieved."""

    CONTENT_UNAVAILABLE = "CONTENT_UNAVAILABLE"
    """Page exists but content couldn't be extracted (auth wall, paywall)."""

    BLOCKED = "BLOCKED"
    """Request was blocked (429, 403, etc.)."""

    ERROR = "ERROR"
    """Technical error during retrieval."""


class EvidenceItem(BaseModel):
    """
    A single piece of evidence mapped to a specific assertion.
    
    CRITICAL DESIGN: Evidence links to assertion_key, not just claim.
    This enables:
    - Per-assertion verification
    - CONTEXT can't refute FACT
    - Granular mixed verdicts
    
    Example:
        EvidenceItem(
            claim_id="c1",
            assertion_key="event.location.city",
            stance=EvidenceStance.SUPPORT,
            excerpt="The fight will take place in Miami...",
            quote="confirmed for Miami Gardens venue",
            is_trusted=True
        )
    """

    claim_id: str
    """ID of the claim this evidence relates to."""

    assertion_key: str = ""
    """
    Which assertion this evidence supports/refutes.
    E.g., "event.location.city", "event.time".
    
    Empty string means evidence applies to whole claim (legacy mode).
    """

    stance: EvidenceStance = EvidenceStance.MENTION
    """Position relative to the assertion. LLM decides."""

    # Content
    excerpt: str = ""
    """Relevant excerpt from the source."""

    quote: str | None = None
    """Direct quote if available (for explainability)."""

    # Source metadata
    domain: str = ""
    """Registrable domain, e.g., "espn.com"."""

    url: str = ""
    """Full URL of the source."""

    title: str = ""
    """Page title."""

    published_at: str | None = None
    """Publication date if available."""

    # Quality signals
    is_primary: bool = False
    """Is this the primary/original source?"""

    is_trusted: bool = False
    """From trusted sources registry."""

    retrieval_confidence: float = Field(default=-1.0, ge=-1.0, le=1.0)
    """
    How confident are we in the retrieval?
    -1.0 = sentinel (not computed yet)
    0.0-1.0 = actual confidence
    """

    relevance_score: float = Field(default=-1.0, ge=-1.0, le=1.0)
    """
    How relevant is this to the assertion?
    -1.0 = sentinel (not computed yet)
    0.0-1.0 = actual relevance
    """

    # M70: Handle unavailable content (THE BUG FIX)
    content_status: ContentStatus = ContentStatus.AVAILABLE
    """
    Was content successfully retrieved?
    
    IMPORTANT: If CONTENT_UNAVAILABLE:
    - Keep in pack for explainability
    - Lower explainability_score
    - Claim stays AMBIGUOUS, not REFUTED
    
    This fixes the OFAC/treasury.gov empty snippet bug.
    """

    unavailable_reason: str | None = None
    """Why content isn't available (for debugging)."""

    # Source classification
    source_type: str = "unknown"
    """primary, official, independent_media, aggregator, social, etc."""

    evidence_tier: str = ""
    """A, A', B, C, D tier classification (if computed)."""

    model_config = {"extra": "ignore"}

    def is_actionable(self) -> bool:
        """Check if this evidence can be used for verdict."""
        return (
            self.content_status == ContentStatus.AVAILABLE
            and self.stance in [EvidenceStance.SUPPORT, EvidenceStance.REFUTE, EvidenceStance.MIXED]
        )

    def affects_explainability(self) -> bool:
        """Check if this evidence affects explainability score."""
        return self.content_status != ContentStatus.AVAILABLE
