"""Evidence Signals - Observable Metrics from Pipeline (Sensors) domain logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


@dataclass
class RetrievalSignals:
    """Signals about source retrieval."""
    total_sources_found: int = 0
    total_sources_considered: int = 0
    total_sources_read: int = 0
    unreadable_sources: int = 0
    unreadable_breakdown: Dict[str, int] = field(default_factory=dict)
    unique_domains_count: int = 0


@dataclass
class CoverageSignals:
    """Signals about assertion coverage."""
    assertions_total: int = 1
    assertions_covered: int = 0
    assertions_with_quotes: int = 0


@dataclass
class TimelinessSignals:
    """Signals about evidence timeliness."""
    newest_source_age_hours: float | None = None
    oldest_source_age_hours: float | None = None


@dataclass
class EvidenceSignals:
    """Complete sensor signals."""
    retrieval: RetrievalSignals = field(default_factory=RetrievalSignals)
    coverage: CoverageSignals = field(default_factory=CoverageSignals)
    timeliness: TimelinessSignals | None = None

    @property
    def has_readable_sources(self) -> bool:
        return self.retrieval.total_sources_read > 0

    @property
    def coverage_ratio(self) -> float:
        return self.coverage.assertions_covered / self.coverage.assertions_total


class TimeGranularity(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    RANGE = "range"
    RELATIVE = "relative"


@dataclass
class TimeWindow:
    """Interpreted time window for a claim."""
    start_date: str | None = None
    end_date: str | None = None
    granularity: TimeGranularity | None = None
    source_signal: str | None = None
    confidence: float | None = None


@dataclass
class LocaleDecision:
    """Recorded locale selection decision for retrieval."""
    primary_locale: str = "en"
    fallback_locales: List[str] = field(default_factory=list)
    used_locales: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    sufficiency_triggered: bool = False

    def to_dict(self) -> dict:
        return {
            "primary_locale": self.primary_locale,
            "fallback_locales": self.fallback_locales,
            "used_locales": self.used_locales,
            "reason_codes": self.reason_codes,
            "sufficiency_triggered": self.sufficiency_triggered,
        }
