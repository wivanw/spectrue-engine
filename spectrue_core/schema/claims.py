# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M70: ClaimUnit and Assertion Pydantic Models

This is the SPEC PRODUCER output format.
Claim Extraction LLM fills these structures - it's a parser, not a judge.

Key Design Principles:
1. Dimension (FACT/CONTEXT/INTERPRETATION) is assigned by LLM, not hardcoded
2. time_reference and location are SEPARATE fields (the core bug fix)
3. Schema is stable - downstream consumers depend on this contract
4. Simple facts work too (ClaimUnit with 1 FACT assertion)
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Type aliases to avoid shadowing
DateType = datetime.date
TimeType = datetime.time


# ─────────────────────────────────────────────────────────────────────────────
# Enums (LLM chooses these, code validates)
# ─────────────────────────────────────────────────────────────────────────────

class Dimension(str, Enum):
    """
    Assertion dimension - determines verification behavior.
    
    LLM decides which dimension applies. Code enforces rules per dimension.
    """
    FACT = "FACT"
    """Must be proven/refuted by evidence. Strict verification."""

    CONTEXT = "CONTEXT"
    """Contextual framing (time zone, audience). Informational only."""

    INTERPRETATION = "INTERPRETATION"
    """Parser interpretation from ambiguous text. Bounded, flagged."""


class VerificationScope(str, Enum):
    """How strictly to verify this assertion."""
    STRICT = "STRICT"
    """For FACT assertions - can be VERIFIED/REFUTED/AMBIGUOUS."""

    SOFT = "SOFT"
    """For CONTEXT - only VERIFIED/AMBIGUOUS unless explicitly contradicted."""


class ClaimDomain(str, Enum):
    """High-level domain of the claim."""
    NEWS = "news"
    SCIENCE = "science"
    POLITICS = "politics"
    FINANCE = "finance"
    HEALTH = "health"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    ENTERTAINMENT = "entertainment"
    HISTORY = "history"
    OTHER = "other"


class ClaimType(str, Enum):
    """Type of claim for search/verification strategy."""
    EVENT = "event"
    """Something happened at a time/place."""

    ATTRIBUTION = "attribution"
    """Someone said/did something."""

    NUMERIC = "numeric"
    """Specific numbers, statistics, measurements."""

    DEFINITION = "definition"
    """What something is/means."""

    COMPARISON = "comparison"
    """X is greater/less/equal to Y."""

    POLICY = "policy"
    """Rules, laws, regulations."""

    TIMELINE = "timeline"
    """Sequence of events, dates."""

    BIOGRAPHY = "biography"
    """Facts about a person."""

    OTHER = "other"


# ─────────────────────────────────────────────────────────────────────────────
# Assertion (Field-Level Fact)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceRequirementSpec(BaseModel):
    """What evidence is required to verify this assertion."""

    needs_primary: bool = False
    """Needs official/primary source confirmation."""

    needs_2_independent: bool = False
    """Needs 2+ independent sources."""

    model_config = {"extra": "ignore"}


class SourceSpan(BaseModel):
    """Location of text in original article."""

    start: int
    """Character offset start."""

    end: int
    """Character offset end."""

    text: str
    """The extracted text span."""

    model_config = {"extra": "ignore"}


class Assertion(BaseModel):
    """
    A single field-level fact within a ClaimUnit.
    
    This is the atomic unit of verification.
    Example: {key: "event.location.city", value: "Miami", dimension: FACT}
    
    The LLM decides dimension classification. Code routes verification accordingly.
    """

    key: str
    """Structured key path. E.g., "event.location.city", "event.time_reference"."""

    value: Any
    """Normalized value (string, number, date, etc.)."""

    value_raw: str | None = None
    """Raw excerpt from article text."""

    dimension: Dimension = Dimension.FACT
    """FACT / CONTEXT / INTERPRETATION. LLM decides."""

    evidence_requirement: EvidenceRequirementSpec = Field(
        default_factory=EvidenceRequirementSpec
    )
    """What evidence is needed."""

    verification_scope: VerificationScope = VerificationScope.STRICT
    """STRICT for FACT, SOFT for CONTEXT."""

    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    """How important is this assertion to the claim? 0-1."""

    is_inferred: bool = False
    """True if LLM inferred this (wasn't explicit in text)."""

    model_config = {"extra": "ignore"}


# ─────────────────────────────────────────────────────────────────────────────
# Qualifier Sub-Schemas
# ─────────────────────────────────────────────────────────────────────────────

class LocationQualifier(BaseModel):
    """
    Structured location information.
    
    CRITICAL: This is SEPARATE from time_reference!
    "(в Україні)" after time is NOT location, it's time context.
    """

    venue: str | None = None
    """E.g., "AT&T Stadium"."""

    city: str | None = None
    """E.g., "Miami"."""

    region: str | None = None
    """E.g., "Florida", "Kyiv Oblast"."""

    country: str | None = None
    """E.g., "USA", "Ukraine"."""

    is_inferred: bool = False
    """True if location was inferred, not explicit in text."""

    model_config = {"extra": "ignore"}


class EventRules(BaseModel):
    """Rules for sports/competition events."""

    max_rounds: int | None = None
    """E.g., 12 for boxing."""

    glove_oz: float | None = None
    """Glove weight in ounces."""

    ring_size_ft: str | None = None
    """Ring dimensions."""

    weight_class: str | None = None
    """E.g., "heavyweight"."""

    model_config = {"extra": "ignore"}


class BroadcastInfo(BaseModel):
    """Broadcast/streaming information."""

    platform: str | None = None
    """E.g., "DAZN", "ESPN+"."""

    start_time_local: str | None = None
    """Local broadcast start time."""

    region_restrictions: str | None = None
    """E.g., "US only", "Global"."""

    model_config = {"extra": "ignore"}


class EventQualifiers(BaseModel):
    """
    Structured qualifiers for event-type claims.
    
    CRITICAL DESIGN: time_reference and location are SEPARATE.
    - time_reference: "Ukraine time", "Kyiv time" → CONTEXT
    - location.city: "Miami" → FACT
    
    This separation eliminates the ambiguity bug.
    """

    # Time dimension
    event_date: DateType | None = None
    """Event date."""

    event_time: TimeType | None = None
    """Event time."""

    datetime_utc: str | None = None
    """ISO datetime in UTC."""

    timezone: str | None = None
    """Timezone name, e.g., "Europe/Kyiv"."""

    time_reference: str | None = None
    """
    Human-readable time context. E.g., "Ukraine time", "Kyiv time".
    
    IMPORTANT: This is CONTEXT, not FACT.
    "(в Україні)" after a time → goes here, NOT to location.
    """

    # Location dimension (SEPARATE from time_reference!)
    location: LocationQualifier | None = None
    """Where the event takes place. This IS a FACT."""

    # Participants
    participants: list[str] = Field(default_factory=list)
    """E.g., ["Anthony Joshua", "Jake Paul"]."""

    # Rules (sports)
    rules: EventRules | None = None
    """Sport-specific rules."""

    # Broadcast
    broadcast: BroadcastInfo | None = None
    """Streaming/TV info."""

    model_config = {"extra": "ignore"}


# ─────────────────────────────────────────────────────────────────────────────
# ClaimUnit (Atomic, Verifiable)
# ─────────────────────────────────────────────────────────────────────────────

class ClaimUnit(BaseModel):
    """
    A structured, schema-grounded claim.
    
    This is the OUTPUT of Claim Extraction (Spec Producer).
    This is the INPUT to Search/Evidence/Scoring (Spec Consumers).
    
    The LLM fills this structure. It's a parser, not a judge.
    It decides dimensions and field values, but doesn't decide truth.
    
    Example (simple fact):
        ClaimUnit(
            claim_type="definition",
            assertions=[Assertion(key="physical.boiling_point", value=100, dimension=FACT)]
        )
    
    Example (complex event):
        ClaimUnit(
            claim_type="event",
            subject="Joshua",
            predicate="scheduled_fight_against",
            object="Paul",
            assertions=[
                Assertion(key="event.time", value="03:00", dimension=FACT),
                Assertion(key="event.time_reference", value="Kyiv time", dimension=CONTEXT),
                Assertion(key="event.location.city", value="Miami", dimension=FACT),
            ]
        )
    """

    id: str
    """Unique claim ID, e.g., "c1"."""

    domain: ClaimDomain = ClaimDomain.OTHER
    """High-level domain: news, science, politics, etc."""

    claim_type: ClaimType = ClaimType.OTHER
    """Type of claim: event, attribution, numeric, etc."""

    # Subject-Predicate-Object structure
    subject: str | None = None
    """The subject of the claim. E.g., "Anthony Joshua"."""

    predicate: str = ""
    """Canonical action/relation. E.g., "scheduled_fight_against"."""

    object: str | None = None
    """The object of the claim. E.g., "Jake Paul"."""

    # Structured qualifiers
    qualifiers: EventQualifiers | None = None
    """Typed qualifiers (time, location, participants, etc.)."""

    # Field-level assertions (THE CORE INNOVATION)
    assertions: list[Assertion] = Field(default_factory=list)
    """
    Each assertion is a field-level fact with dimension.
    FACT assertions are strictly verified.
    CONTEXT assertions are informational.
    
    This list is what downstream consumers iterate over.
    """

    # Metadata
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    """How important is this claim? 0-1."""

    check_worthiness: float = Field(default=0.5, ge=0.0, le=1.0)
    """Is this worth checking? 0-1. Filters out opinions."""

    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    """How confident is the extractor in this structure? 0-1."""

    source_span: SourceSpan | None = None
    """Location in original text."""

    language: str = "en"
    """Language of the claim text."""

    # Legacy compatibility fields
    text: str = ""
    """Legacy: raw claim text. Prefer structured fields."""

    normalized_text: str = ""
    """Legacy: self-sufficient text with context. Prefer structured fields."""

    topic_group: str = "Other"
    """Legacy: topic category."""

    topic_key: str = ""
    """Legacy: specific entity tag for round-robin."""

    model_config = {"extra": "ignore"}

    def get_fact_assertions(self) -> list[Assertion]:
        """Get only FACT assertions (for strict verification)."""
        return [a for a in self.assertions if a.dimension == Dimension.FACT]

    def get_context_assertions(self) -> list[Assertion]:
        """Get only CONTEXT assertions (informational)."""
        return [a for a in self.assertions if a.dimension == Dimension.CONTEXT]

    def has_location(self) -> bool:
        """Check if claim has explicit location (FACT)."""
        if self.qualifiers and self.qualifiers.location:
            loc = self.qualifiers.location
            return any([loc.venue, loc.city, loc.region, loc.country])
        return any(
            a.key.startswith("event.location") and a.dimension == Dimension.FACT
            for a in self.assertions
        )
