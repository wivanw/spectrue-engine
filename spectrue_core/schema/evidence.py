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
EvidenceItem Pydantic Model

Evidence is a DOWNSTREAM CONSUMER of the schema.
It maps to specific assertion_key, not just claim_id.

Key Design Principles:
1. Evidence links to assertion_key (not just raw claim text)
2. CONTENT_UNAVAILABLE status is explicit (not silent drop)
3. Stance is per-assertion, enabling granular verification
"""

from enum import Enum
from typing import Literal

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel


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

    CONTEXT = "CONTEXT"
    """Additional context preserved for data retention."""


class TimelinessStatus(str, Enum):
    """Timeliness of evidence relative to claim time window."""

    IN_WINDOW = "in_window"
    OUTDATED = "outdated"
    UNKNOWN_DATE = "unknown_date"


class EvidenceNeedType(str, Enum):
    """
    Layer 4: Evidence type classification for routing.
    
    Used ONLY to route search strategies and sources.
    Does NOT assert existence or quality of evidence.
    """
    EMPIRICAL_STUDY = "empirical_study"
    """Requires scientific research, clinical trials, peer-reviewed studies."""

    GUIDELINE = "guideline"
    """Requires official guidelines, consensus statements, policy documents."""

    OFFICIAL_STATS = "official_stats"
    """Requires government statistics, census data, official reports."""

    EXPERT_OPINION = "expert_opinion"
    """Requires expert quotes, professional assessments."""

    ANECDOTAL = "anecdotal"
    """Personal testimonies, case studies only."""

    NEWS_REPORT = "news_report"
    """Requires journalistic coverage of events."""

    UNKNOWN = "unknown"
    """Cannot determine what evidence is needed."""


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


class EvidenceItem(SchemaModel):
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

    claim_id: str | None
    """ID of the claim this evidence relates to (None if purely contextual)."""

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

    timeliness_status: TimelinessStatus = TimelinessStatus.UNKNOWN_DATE
    """Timeliness relative to the claim's time window."""

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

    # Handle unavailable content (THE BUG FIX)
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

    # --- Evidence metadata (typed, non-heuristic) ---
    evidence_role: Literal["direct", "indirect", "mention_only"] = "indirect"
    """Classification of the evidence's role in supporting/refuting the claim."""

    # What parts of the claim this evidence covers
    covers: list[
        Literal[
            "entity",
            "time",
            "location",
            "quantity",
            "attribution",
            "causal",
            "other",
        ]
    ] = []
    """List of claim components (facets) that this evidence explicitly addresses."""

    # Event compatibility signature (for routing)
    event_signature: dict | None = None
    """Semantic signature for event-level search and routing."""

    # Provenance
    provenance: Literal["direct", "transferred"] = "direct"
    """Original source vs evidence shared/transferred from another claim."""

    origin_claim_id: str | None = None
    """Original claim_id if this evidence was transferred."""

    # --- Dedup / corroboration metadata ---
    publisher_id: str = ""          # normalized domain/publisher id
    content_hash: str = ""          # sha256 of normalized text payload (exact dup group)
    similar_cluster_id: str = ""    # simhash bucket id (near-dup cluster)

    def is_actionable(self) -> bool:
        """Check if this evidence can be used for verdict."""
        return (
            self.content_status == ContentStatus.AVAILABLE
            and self.stance in [EvidenceStance.SUPPORT, EvidenceStance.REFUTE, EvidenceStance.MIXED]
        )

    def affects_explainability(self) -> bool:
        """Check if this evidence affects explainability score."""
        return self.content_status != ContentStatus.AVAILABLE
