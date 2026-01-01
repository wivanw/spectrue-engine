# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Execution Plan Types

Defines the structure for progressive widening search execution:
- Phase: Single search iteration (locale + channels + depth)
- ExecutionPlan: Complete plan for all claims
- BudgetClass: Controls how many phases to execute

Waterfall Logic:
- Phase A: Primary locale, high-quality channels, basic depth, k=3
- Phase B: Primary locale, + local media, advanced depth, k=5
- Phase C: Fallback locale, high-quality channels, basic depth, k=3
- Phase D: All channels, advanced depth, k=7 (last resort)

Features:
- Early exit on sufficiency (stop before later phases)
- Parallel execution within phase (across claims)
- Sequential execution across phases (waterfall)
- Fail-open: Phase A-light for low confidence claims
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spectrue_core.schema.claim_metadata import EvidenceChannel, UsePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Budget Class
# ─────────────────────────────────────────────────────────────────────────────

class BudgetClass(str, Enum):
    """
    Search budget classification.
    
    Controls how many phases are included in ExecutionPlan.
    Higher budget = more phases = higher cost = better coverage.
    """
    MINIMAL = "minimal"
    """Only Phase A. Fastest, cheapest. For low-priority claims."""

    STANDARD = "standard"
    """Phases A + B. Balanced cost/coverage. Default."""

    DEEP = "deep"
    """All phases (A/B/C/D). Maximum coverage. For high-priority claims."""


class PolicyMode(str, Enum):
    """Per-claim policy decision for search routing."""
    SKIP = "SKIP"
    CHEAP = "CHEAP"
    FULL = "FULL"


@dataclass(frozen=True)
class ClaimPolicyDecision:
    """Routing decision result for a claim before query building."""
    mode: PolicyMode
    reason_codes: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Phase Definition
# ─────────────────────────────────────────────────────────────────────────────

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

    max_results: int = 3
    """Maximum results to retrieve (k parameter)."""

    is_expensive: bool = False
    """Cost flag. True for advanced depth or large k."""

    description: str = ""
    """Human-readable description for tracing."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tracing/logging."""
        return {
            "phase_id": self.phase_id,
            "locale": self.locale,
            "channels": [c.value for c in self.channels],
            "use_policy_by_channel": {k: v.value for k, v in (self.use_policy_by_channel or {}).items()},
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "is_expensive": self.is_expensive,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase Presets
# ─────────────────────────────────────────────────────────────────────────────

def phase_a(locale: str) -> Phase:
    """
    Phase A: Primary locale, high-quality channels, basic depth.
    
    This is the first and most important phase.
    If sufficient evidence is found here, later phases are skipped.
    """
    return Phase(
        phase_id="A",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth="basic",
        max_results=3,
        is_expensive=False,
        description=f"Primary search: {locale}, authoritative+reputable, k=3",
    )


def phase_a_light(locale: str = "en") -> Phase:
    """
    Phase A-light: Minimal search for fail-open scenarios.
    
    Used when:
    - metadata_confidence == LOW
    - verification_target == NONE but fail-open applies
    
    Very cheap: only authoritative, k=2.
    """
    return Phase(
        phase_id="A-light",
        locale=locale,
        channels=[EvidenceChannel.AUTHORITATIVE],
        search_depth="basic",
        max_results=2,
        is_expensive=False,
        description=f"Fail-open minimal: {locale}, authoritative only, k=2",
    )


def phase_a_origin(locale: str) -> Phase:
    """
    Phase A-origin: Origin-focused search for attribution claims.
    
    Prioritizes finding the original source of a quote/statement.
    """
    return Phase(
        phase_id="A-origin",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth="basic",
        max_results=3,
        is_expensive=False,
        description=f"Origin search: {locale}, finding original source",
    )


def phase_b(locale: str) -> Phase:
    """
    Phase B: Primary locale, expanded channels, advanced depth.
    
    Includes local media for broader coverage.
    """
    return Phase(
        phase_id="B",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
            EvidenceChannel.LOCAL_MEDIA,
        ],
        search_depth="advanced",
        max_results=5,
        is_expensive=True,
        description=f"Expanded search: {locale}, +local media, k=5",
    )


def phase_c(locale: str) -> Phase:
    """
    Phase C: Fallback locale, high-quality channels.
    
    Used when primary locale yields insufficient evidence.
    Typically locale is from search_locale_plan.fallback.
    """
    return Phase(
        phase_id="C",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth="basic",
        max_results=3,
        is_expensive=False,
        description=f"Fallback locale search: {locale}, k=3",
    )


def phase_d(locale: str = "en") -> Phase:
    """
    Phase D: Last resort, all channels, maximum depth.
    
    Includes social and low-reliability as leads.
    This is expensive and should only run if earlier phases fail.
    """
    return Phase(
        phase_id="D",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
            EvidenceChannel.LOCAL_MEDIA,
            EvidenceChannel.SOCIAL,
            EvidenceChannel.LOW_RELIABILITY,
        ],
        search_depth="advanced",
        max_results=7,
        is_expensive=True,
        description=f"Last resort: {locale}, all channels, k=7",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execution Plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    """
    Complete execution plan for all claims.
    
    Maps each claim_id to its list of phases to execute.
    Phases are executed in order; early exit on sufficiency.
    
    Example:
        plan = ExecutionPlan(
            claim_phases={
                "c1": [phase_a("en"), phase_b("en")],  # Fact claim
                "c2": [phase_a_light("en")],           # Prediction (fail-open)
            },
            budget_class=BudgetClass.STANDARD
        )
    """
    claim_phases: dict[str, list[Phase]] = field(default_factory=dict)
    """Mapping of claim_id to list of phases."""

    budget_class: BudgetClass = BudgetClass.STANDARD
    """Budget classification used to build this plan."""

    # M113: Pipeline profile metadata
    profile_name: str | None = None
    """Name of the pipeline profile used to build this plan (M113)."""

    profile_version: str | None = None
    """Version of the pipeline profile (M113)."""

    overrides: dict[str, Any] | None = None
    """Per-run overrides applied to the profile (M113)."""

    max_credits: int | None = None
    """Maximum credits for this run (M113)."""

    def get_phases(self, claim_id: str) -> list[Phase]:
        """Get phases for a claim. Returns empty list if not found."""
        return self.claim_phases.get(claim_id, [])

    def add_claim(self, claim_id: str, phases: list[Phase]) -> None:
        """Add phases for a claim."""
        self.claim_phases[claim_id] = phases

    def get_all_phase_ids(self) -> set[str]:
        """Get all unique phase IDs in the plan."""
        phase_ids: set[str] = set()
        for phases in self.claim_phases.values():
            for phase in phases:
                phase_ids.add(phase.phase_id)
        return phase_ids

    def get_claims_needing_phase(self, phase_id: str) -> list[str]:
        """Get list of claim IDs that need a specific phase."""
        return [
            claim_id
            for claim_id, phases in self.claim_phases.items()
            if any(p.phase_id == phase_id for p in phases)
        ]

    @property
    def total_phases(self) -> int:
        """Total number of phase executions across all claims."""
        return sum(len(phases) for phases in self.claim_phases.values())

    @property
    def max_depth(self) -> int:
        """Maximum number of phases for any single claim."""
        if not self.claim_phases:
            return 0
        return max(len(phases) for phases in self.claim_phases.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tracing/logging."""
        result = {
            "budget_class": self.budget_class.value,
            "total_phases": self.total_phases,
            "max_depth": self.max_depth,
            "claim_phases": {
                claim_id: [p.to_dict() for p in phases]
                for claim_id, phases in self.claim_phases.items()
            },
        }
        # M113: Include profile metadata if present
        if self.profile_name:
            result["profile_name"] = self.profile_name
        if self.profile_version:
            result["profile_version"] = self.profile_version
        if self.overrides:
            result["overrides"] = self.overrides
        if self.max_credits is not None:
            result["max_credits"] = self.max_credits
        return result

    def summary(self) -> str:
        """Human-readable summary."""
        header = f"ExecutionPlan (budget={self.budget_class.value}"
        if self.profile_name:
            header += f", profile={self.profile_name}"
        if self.max_credits is not None:
            header += f", max_credits={self.max_credits}"
        header += "):"
        lines = [header]
        for claim_id, phases in self.claim_phases.items():
            phase_ids = [p.phase_id for p in phases]
            lines.append(f"  {claim_id}: {' → '.join(phase_ids)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Execution State (Runtime)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalHop:
    """A single retrieval hop for a claim."""
    hop_index: int
    query: str
    locale: str
    channels: list[EvidenceChannel]
    search_depth: str
    results_count: int
    decision: str
    decision_reason: str
    cost_credits: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "query": self.query,
            "locale": self.locale,
            "channels": [c.value for c in self.channels],
            "search_depth": self.search_depth,
            "results_count": self.results_count,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "cost_credits": self.cost_credits,
        }


@dataclass
class ClaimExecutionState:
    """
    Runtime state for a claim's execution.
    
    Tracks which phases have completed and whether sufficiency was reached.
    Updated by PhaseRunner during execution.
    """
    claim_id: str
    """The claim being executed."""

    phases_completed: list[str] = field(default_factory=list)
    """Phase IDs that have completed."""

    hops: list[RetrievalHop] = field(default_factory=list)
    """Retrieval hops executed for this claim."""

    is_sufficient: bool = False
    """True if sufficient evidence found (stop early)."""

    sufficiency_reason: str = ""
    """Why sufficiency was reached (for logging)."""

    stop_reason: str | None = None
    """Optional stop reason for execution termination."""

    phases_skipped: list[str] = field(default_factory=list)
    """Phase IDs skipped due to sufficiency."""

    error: str | None = None
    """Error message if execution failed."""

    def mark_completed(self, phase_id: str) -> None:
        """Mark a phase as completed."""
        if phase_id not in self.phases_completed:
            self.phases_completed.append(phase_id)

    def mark_sufficient(self, reason: str, remaining_phases: list[str]) -> None:
        """Mark as sufficient and record skipped phases."""
        self.is_sufficient = True
        self.sufficiency_reason = reason
        self.stop_reason = "sufficiency_met"
        self.phases_skipped = remaining_phases

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tracing."""
        return {
            "claim_id": self.claim_id,
            "phases_completed": self.phases_completed,
            "hops": [hop.to_dict() for hop in self.hops],
            "is_sufficient": self.is_sufficient,
            "sufficiency_reason": self.sufficiency_reason,
            "stop_reason": self.stop_reason,
            "phases_skipped": self.phases_skipped,
            "error": self.error,
        }


@dataclass
class ExecutionState:
    """
    Overall execution state for all claims.
    
    Aggregates ClaimExecutionState for each claim.
    """
    claim_states: dict[str, ClaimExecutionState] = field(default_factory=dict)
    """State per claim_id."""

    def get_or_create(self, claim_id: str) -> ClaimExecutionState:
        """Get or create state for a claim."""
        if claim_id not in self.claim_states:
            self.claim_states[claim_id] = ClaimExecutionState(claim_id=claim_id)
        return self.claim_states[claim_id]

    def all_sufficient(self) -> bool:
        """Check if all claims have sufficient evidence."""
        if not self.claim_states:
            return False
        return all(s.is_sufficient for s in self.claim_states.values())

    def claims_needing_more(self) -> list[str]:
        """Get claim IDs that still need more phases."""
        return [
            claim_id
            for claim_id, state in self.claim_states.items()
            if not state.is_sufficient and state.error is None
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tracing."""
        return {
            claim_id: state.to_dict()
            for claim_id, state in self.claim_states.items()
        }
