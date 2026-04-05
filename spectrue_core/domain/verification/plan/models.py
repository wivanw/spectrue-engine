"""
Core models for execution planning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spectrue_core.domain.evidence.model import EvidenceChannel, UsePolicy


class BudgetClass(str, Enum):
    """
    Search budget classification.
    
    Controls how many phases are included in ExecutionPlan.
    """
    MINIMAL = "minimal"
    """Only Phase A. Fastest, cheapest. For low-priority claims."""

    BALANCED = "balanced"
    """Phases A + B. Balanced cost/coverage. Default."""

    COMPREHENSIVE = "comprehensive"
    """All phases (A/B/C/D). Maximum coverage. For high-priority claims."""


@dataclass
class Phase:
    """
    A single search phase in the progressive widening waterfall.
    
    Each phase represents one search iteration with specific:
    - Locale (language/region for search)
    - Channels (which source tiers to include)
    - Depth (basic vs advanced search)
    - Result limit (k parameter)
    
    Phases are executed sequentially per claim, with early exit on sufficiency.
    """
    phase_id: str
    """Phase identifier: 'A', 'B', 'C', 'D', 'A-light', 'A-origin'."""

    locale: str
    """Search locale/language. E.g., 'en', 'uk', 'de'."""

    channels: list[EvidenceChannel]
    """Which source channels to search."""

    use_policy_by_channel: dict[str, UsePolicy] = field(default_factory=dict)
    """
    Per-channel usage policy for this phase (support_ok vs lead_only).
    
    This mirrors `RetrievalPolicy.use_policy_by_channel` but is scoped to the
    channels present in this phase.
    """

    search_depth: str = "basic"
    """Search depth: 'basic' or 'advanced'. Maps to Tavily depth."""

    max_results: int = 5
    """Maximum results per query (k parameter)."""

    is_expensive: bool = False
    """Whether this phase is considered expensive (e.g. advanced search)."""

    description: str = ""
    """Human-readable description for tracing."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize phase configuration."""
        return {
            "phase_id": self.phase_id,
            "locale": self.locale,
            "channels": [c.value for c in self.channels],
            "use_policy_by_channel": {k: v.value for k, v in self.use_policy_by_channel.items()},
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "is_expensive": self.is_expensive,
            "description": self.description,
        }
