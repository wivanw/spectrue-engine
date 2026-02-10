# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.domain.evidence.model import EvidenceChannel
    from spectrue_core.domain.verification.plan import Phase, ExecutionPlan

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
            "channels": [c.value if hasattr(c, 'value') else str(c) for c in self.channels],
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
    """
    claim_id: str
    phases_completed: list[str] = field(default_factory=list)
    hops: list[RetrievalHop] = field(default_factory=list)
    is_sufficient: bool = False
    sufficiency_reason: str = ""
    stop_reason: str | None = None
    phases_skipped: list[str] = field(default_factory=list)
    error: str | None = None

    def mark_completed(self, phase_id: str) -> None:
        if phase_id not in self.phases_completed:
            self.phases_completed.append(phase_id)

    def mark_sufficient(self, reason: str, remaining_phases: list[str]) -> None:
        self.is_sufficient = True
        self.sufficiency_reason = reason
        self.stop_reason = "sufficiency_met"
        self.phases_skipped = remaining_phases

    def to_dict(self) -> dict[str, Any]:
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

    def build_trace_summary(
        self,
        *,
        phases: list[Phase] | None = None,
        llm_calls_count: int = 0,
    ) -> dict[str, Any]:
        phases_executed = list(self.phases_completed)
        channels_used: list[str] = []

        phase_map = {p.phase_id: p for p in (phases or [])}
        for phase_id in phases_executed:
            phase = phase_map.get(phase_id)
            if not phase:
                continue
            for channel in phase.channels or []:
                value = channel.value if hasattr(channel, "value") else str(channel)
                if value not in channels_used:
                    channels_used.append(value)

        if not channels_used:
            for hop in self.hops:
                if not hop.channels:
                    continue
                for channel in hop.channels:
                    value = channel.value if hasattr(channel, "value") else str(channel)
                    if value not in channels_used:
                        channels_used.append(value)

        total_results = 0
        tavily_calls = 0
        for hop in self.hops:
            tavily_calls += 1
            total_results += hop.results_count

        return {
            "claim_id": self.claim_id,
            "llm_calls_count": int(llm_calls_count),
            "tavily_calls_count": int(tavily_calls),
            "total_search_results": int(total_results),
            "phases_executed": phases_executed,
            "channels_used": channels_used,
        }


@dataclass
class ExecutionState:
    """
    Overall execution state for all claims.
    """
    claim_states: dict[str, ClaimExecutionState] = field(default_factory=dict)

    def get_or_create(self, claim_id: str) -> ClaimExecutionState:
        if claim_id not in self.claim_states:
            self.claim_states[claim_id] = ClaimExecutionState(claim_id=claim_id)
        return self.claim_states[claim_id]

    def all_sufficient(self) -> bool:
        if not self.claim_states:
            return False
        return all(s.is_sufficient for s in self.claim_states.values())

    def claims_needing_more(self) -> list[str]:
        return [
            claim_id
            for claim_id, state in self.claim_states.items()
            if not state.is_sufficient and state.error is None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            claim_id: state.to_dict()
            for claim_id, state in self.claim_states.items()
        }

    def build_trace_summaries(
        self,
        *,
        plan: ExecutionPlan,
        llm_calls_by_claim: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for claim_id, state in self.claim_states.items():
            summaries.append(
                state.build_trace_summary(
                    phases=plan.get_phases(claim_id),
                    llm_calls_count=(llm_calls_by_claim or {}).get(claim_id, 0),
                )
            )
        return summaries
