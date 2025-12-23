# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M71: Evidence Signals - Observable Metrics from Pipeline (Sensors).
Pragmatic strictness: critical invariants only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from enum import Enum

from pydantic import Field, model_validator

from spectrue_core.schema.serialization import SchemaModel

if TYPE_CHECKING:
    from typing import Self


class RetrievalSignals(SchemaModel):
    """Signals about source retrieval."""
    
    total_sources_found: int = Field(default=0, ge=0)
    total_sources_considered: int = Field(default=0, ge=0)
    total_sources_read: int = Field(default=0, ge=0)
    
    # Granular tracking (M71 requirement)
    unreadable_sources: int = Field(default=0, ge=0)
    unreadable_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    unique_domains_count: int = Field(default=0, ge=0)
    
    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        if self.total_sources_considered > self.total_sources_found:
            raise ValueError("Considered > Found")
            
        if self.total_sources_read + self.unreadable_sources > self.total_sources_considered:
            raise ValueError("Read + Unreadable > Considered")
            
        if self.total_sources_considered > 0 and self.unique_domains_count == 0:
            raise ValueError("Sources considered but 0 unique domains found")
             
        return self
    

class CoverageSignals(SchemaModel):
    """Signals about assertion coverage."""
    
    # Default is 1 to avoid division by zero, but pipeline must set true value.
    assertions_total: int = Field(default=1, ge=1)
    assertions_covered: int = Field(default=0, ge=0)
    assertions_with_quotes: int = Field(default=0, ge=0)
    
    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        if self.assertions_covered > self.assertions_total:
            raise ValueError("Covered > Total")
        if self.assertions_with_quotes > self.assertions_covered:
            raise ValueError("Quotes > Covered")
        return self
    

class TimelinessSignals(SchemaModel):
    """Signals about evidence timeliness."""
    
    newest_source_age_hours: float | None = Field(default=None, ge=0.0)
    oldest_source_age_hours: float | None = Field(default=None, ge=0.0)
    
    @model_validator(mode="after")
    def validate_chronology(self) -> Self:
        if (
            self.newest_source_age_hours is not None
            and self.oldest_source_age_hours is not None
            and self.newest_source_age_hours > self.oldest_source_age_hours
        ):
            raise ValueError("Newest age > Oldest age")
        return self
    

class EvidenceSignals(SchemaModel):
    """Complete sensor signals."""
    
    retrieval: RetrievalSignals = Field(default_factory=RetrievalSignals)
    coverage: CoverageSignals = Field(default_factory=CoverageSignals)
    timeliness: TimelinessSignals | None = Field(default=None)
    
    @property
    def has_readable_sources(self) -> bool:
        """True if at least one source was successfully read."""
        return self.retrieval.total_sources_read > 0
    
    @property
    def coverage_ratio(self) -> float:
        """Fraction of assertions covered (0-1). Safe division (total >= 1)."""
        return self.coverage.assertions_covered / self.coverage.assertions_total


class TimeGranularity(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    RANGE = "range"
    RELATIVE = "relative"


class TimeWindow(SchemaModel):
    """Interpreted time window for a claim."""

    start_date: str | None = None
    end_date: str | None = None
    granularity: TimeGranularity | None = None
    source_signal: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LocaleDecision(SchemaModel):
    """Recorded locale selection decision for retrieval."""

    primary_locale: str
    fallback_locales: List[str] = Field(default_factory=list)
    used_locales: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    sufficiency_triggered: bool = False
