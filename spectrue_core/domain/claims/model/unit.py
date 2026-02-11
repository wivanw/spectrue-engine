"""
Claim unit model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .enums import ClaimRole, ClaimDomain, ClaimType, Dimension
from .structure import ClaimStructure, EventQualifiers, Assertion, SourceSpan


@dataclass
class ClaimUnit:
    """A structured, schema-grounded claim."""
    id: str
    domain: ClaimDomain = ClaimDomain.OTHER
    claim_type: ClaimType = ClaimType.OTHER
    claim_role: ClaimRole = ClaimRole.CORE
    structure: ClaimStructure | None = None
    subject: str | None = None
    predicate: str = ""
    object: str | None = None
    qualifiers: EventQualifiers | None = None
    assertions: list[Assertion] = field(default_factory=list)
    importance: float = 1.0
    check_worthiness: float = 0.5
    extraction_confidence: float = 1.0
    source_span: SourceSpan | None = None
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
