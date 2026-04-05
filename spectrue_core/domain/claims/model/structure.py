"""
Structured claim components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Dimension, VerificationScope, ClaimStructureType


@dataclass
class ClaimStructure:
    """Structured representation of claim logic."""
    type: ClaimStructureType = ClaimStructureType.OTHER
    premises: list[str] = field(default_factory=list)
    conclusion: str | None = None
    dependencies: list[str] = field(default_factory=list)


@dataclass
class EvidenceRequirementSpec:
    """What evidence is required to verify this assertion."""
    needs_primary: bool = False
    needs_2_independent: bool = False


@dataclass
class Assertion:
    """A single field-level fact within a ClaimUnit."""
    key: str
    value: Any
    dimension: Dimension = Dimension.FACT
    evidence_requirement: EvidenceRequirementSpec = field(default_factory=EvidenceRequirementSpec)
    verification_scope: VerificationScope = VerificationScope.STRICT
    importance: float = 1.0


@dataclass
class LocationQualifier:
    """Structured location information."""
    venue: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    is_inferred: bool = False


@dataclass
class BroadcastInfo:
    """Broadcast/streaming information."""
    platform: str | None = None


@dataclass
class EventQualifiers:
    """Structured qualifiers for event-type claims."""
    event_date: Any | None = None
    event_time: Any | None = None
    datetime_utc: str | None = None
    timezone: str | None = None
    time_reference: str | None = None
    location: LocationQualifier | None = None
    participants: list[str] = field(default_factory=list)
    broadcast: BroadcastInfo | None = None
