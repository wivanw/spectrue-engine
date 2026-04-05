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
ClaimUnit and Assertion Pydantic Models (Schema Adapter)

This module imports domain types and exposes them as schema types.
"""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel
from spectrue_core.schema.claim_metadata import ClaimRole
from spectrue_core.domain.claims.model import (
    Dimension,
    VerificationScope,
    ClaimDomain,
    ClaimType,
    ClaimStructureType,
    ClaimStructure as DomainClaimStructure,
    EvidenceRequirementSpec as DomainEvidenceRequirementSpec,
    Assertion as DomainAssertion,
)

# Re-export enums
__all__ = [
    "Dimension",
    "VerificationScope",
    "ClaimDomain",
    "ClaimType",
    "ClaimStructureType",
    "ClaimStructure",
    "EvidenceRequirementSpec",
    "Assertion",
    "LocationQualifier",
    "BroadcastInfo",
    "EventQualifiers",
    "ClaimUnit",
]

# Type aliases to avoid shadowing
DateType = datetime.date
TimeType = datetime.time


class ClaimStructure(SchemaModel, DomainClaimStructure):
    """Structured representation of claim logic."""
    type: ClaimStructureType = ClaimStructureType.OTHER
    premises: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    dependencies: list[str] = Field(default_factory=list)


class EvidenceRequirementSpec(SchemaModel, DomainEvidenceRequirementSpec):
    """What evidence is required to verify this assertion."""
    needs_primary: bool = False
    needs_2_independent: bool = False


class Assertion(SchemaModel, DomainAssertion):
    """
    A single field-level fact within a ClaimUnit.
    """
    key: str
    value: Any
    dimension: Dimension = Dimension.FACT
    evidence_requirement: EvidenceRequirementSpec = Field(
        default_factory=EvidenceRequirementSpec
    )
    verification_scope: VerificationScope = VerificationScope.STRICT
    importance: float = Field(default=1.0, ge=0.0, le=1.0)


class LocationQualifier(SchemaModel):
    """
    Structured location information.
    """
    venue: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    is_inferred: bool = False


class BroadcastInfo(SchemaModel):
    """Broadcast/streaming information."""
    platform: str | None = None


class EventQualifiers(SchemaModel):
    """
    Structured qualifiers for event-type claims.
    """
    event_date: DateType | None = None
    event_time: TimeType | None = None
    datetime_utc: str | None = None
    timezone: str | None = None
    time_reference: str | None = None
    location: LocationQualifier | None = None
    participants: list[str] = Field(default_factory=list)
    broadcast: BroadcastInfo | None = None


class ClaimUnit(SchemaModel):
    """
    A structured, schema-grounded claim.
    """
    id: str
    domain: ClaimDomain = ClaimDomain.OTHER
    claim_type: ClaimType = ClaimType.OTHER
    claim_role: ClaimRole = ClaimRole.CORE
    structure: ClaimStructure | None = None
    subject: str | None = None
    predicate: str = ""
    object: str | None = None
    qualifiers: EventQualifiers | None = None
    assertions: list[Assertion] = Field(default_factory=list)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    check_worthiness: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = "en"
    text: str = ""
    normalized_text: str = ""
    topic_group: str = "Other"
    topic_key: str = ""

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
