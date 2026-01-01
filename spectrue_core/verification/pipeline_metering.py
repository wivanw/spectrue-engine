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
Pipeline Metering

Cost tracking, ledger management, and phase timing utilities.
Extracted from ValidationPipeline.execute() for better modularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from spectrue_core import PROMPT_VERSION, SEARCH_STRATEGY_VERSION, __version__
from spectrue_core.billing.config_loader import load_pricing_policy
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.metering import LLMMeter, TavilyMeter
from spectrue_core.billing.progress_emitter import CostProgressEmitter
from spectrue_core.utils.trace import Trace, current_trace_id
from spectrue_core.verification.costs import (
    map_stage_costs_to_phases,
    summarize_reason_codes,
)
from spectrue_core.verification.ledger_models import (
    BudgetAllocation,
    ClaimLedgerEntry,
    PhaseUsage,
    PipelineCounts,
    ReasonSummary,
    RunLedger,
)
from spectrue_core.verification.ledger_models import (
    ReasonCode as LedgerReasonCode,
)


@dataclass
class PhaseTracker:
    """
    Tracks phase timings and reason codes during pipeline execution.
    """
    phase_timings: dict[str, int] = field(default_factory=dict)
    phase_reason_codes: dict[str, list[LedgerReasonCode]] = field(default_factory=dict)
    reason_events: list[dict] = field(default_factory=list)
    claim_entries: dict[str, ClaimLedgerEntry] = field(default_factory=dict)

    def start_phase(self, phase: str) -> None:
        """Start timing a phase."""
        Trace.phase_start(phase)

    def end_phase(self, phase: str) -> None:
        """End timing a phase and record duration."""
        duration = Trace.phase_end(phase)
        if duration is not None:
            self.phase_timings[phase] = int(duration)

    def record_reason(
        self,
        spec: Any,
        *,
        claim_id: str | None = None,
        count: int = 1,
        sc_cost: float = 0.0,
        tc_cost: float = 0.0,
    ) -> LedgerReasonCode:
        """Record a reason code event."""
        code = spec.qualified()
        entry = LedgerReasonCode(
            code=code,
            label=spec.label,
            phase=spec.phase,
            action=spec.action,
            count=count,
        )
        self.phase_reason_codes.setdefault(spec.phase, []).append(entry)
        self.reason_events.append(
            {
                "code": code,
                "label": spec.label,
                "phase": spec.phase,
                "action": spec.action,
                "count": count,
                "sc_cost": sc_cost,
                "tc_cost": tc_cost,
            }
        )
        Trace.reason_code(
            code=code,
            phase=spec.phase,
            action=spec.action,
            label=spec.label,
            claim_id=claim_id,
        )
        return entry


@dataclass
class MeteringContext:
    """
    Holds all metering-related state for a pipeline run.
    """
    ledger: CostLedger
    tavily_meter: TavilyMeter
    llm_meter: LLMMeter
    progress_emitter: CostProgressEmitter
    run_ledger: RunLedger
    phase_tracker: PhaseTracker


def create_metering_context() -> MeteringContext:
    """
    Create a new metering context for a pipeline run.

    Returns:
        MeteringContext with all metering infrastructure initialized
    """
    policy = load_pricing_policy()
    ledger = CostLedger(run_id=current_trace_id())
    tavily_meter = TavilyMeter(ledger=ledger, policy=policy)
    llm_meter = LLMMeter(ledger=ledger, policy=policy)

    progress_emitter = CostProgressEmitter(
        ledger=ledger,
        min_delta_to_show=policy.min_delta_to_show,
        emit_cost_deltas=policy.emit_cost_deltas,
    )

    run_ledger = RunLedger(
        run_id=current_trace_id(),
        engine_version=__version__,
        prompt_version=PROMPT_VERSION,
        search_strategy=SEARCH_STRATEGY_VERSION,
        counts=PipelineCounts(),
    )

    phase_tracker = PhaseTracker()

    return MeteringContext(
        ledger=ledger,
        tavily_meter=tavily_meter,
        llm_meter=llm_meter,
        progress_emitter=progress_emitter,
        run_ledger=run_ledger,
        phase_tracker=phase_tracker,
    )


def compute_budget_allocation(metadata: Any) -> BudgetAllocation:
    """
    Compute budget allocation based on claim metadata.

    Args:
        metadata: Claim metadata with check_worthiness field

    Returns:
        BudgetAllocation with tier and limits
    """
    worthiness = float(getattr(metadata, "check_worthiness", 0.5) or 0.5)
    if worthiness >= 0.75:
        return BudgetAllocation(
            worthiness_tier="high",
            max_queries=3,
            max_docs=8,
            max_escalations=2,
            defer_allowed=False,
        )
    if worthiness <= 0.35:
        return BudgetAllocation(
            worthiness_tier="low",
            max_queries=1,
            max_docs=3,
            max_escalations=0,
            defer_allowed=True,
        )
    return BudgetAllocation(
        worthiness_tier="medium",
        max_queries=2,
        max_docs=5,
        max_escalations=1,
        defer_allowed=False,
    )


async def create_progress_callback(
    progress_callback: Callable[[str, int | None, int | None], Awaitable[None]] | None,
    progress_emitter: CostProgressEmitter,
) -> Callable[[str, int | None, int | None], Awaitable[None]]:
    """
    Create a wrapped progress callback that also emits cost deltas.

    Args:
        progress_callback: User-provided progress callback (optional)
        progress_emitter: CostProgressEmitter instance

    Returns:
        Async progress callback function
    """
    async def _progress(stage: str, processed: int | None = None, total: int | None = None) -> None:
        if progress_callback:
            await progress_callback(stage, processed, total)
        payload = progress_emitter.maybe_emit(stage=stage)
        if payload:
            Trace.progress_cost_delta(
                stage=payload.stage,
                delta=payload.delta,
                total=payload.total,
            )
    return _progress


def attach_cost_summary(
    payload: dict,
    *,
    metering: MeteringContext,
) -> dict:
    """
    Attach cost summary to the result payload.

    Args:
        payload: Result dict to augment
        metering: MeteringContext with ledger and tracker

    Returns:
        Augmented payload with cost_summary and related fields
    """
    ledger = metering.ledger
    run_ledger = metering.run_ledger
    phase_tracker = metering.phase_tracker

    summary_obj = ledger.get_summary()
    phase_costs = map_stage_costs_to_phases(summary_obj.by_stage_credits)
    phase_order = [
        "extraction",
        "graph",
        "query_build",
        "retrieval",
        "evidence_eval",
        "verdict",
    ]
    run_ledger.phase_usage = [
        PhaseUsage(
            phase=phase,
            sc_cost=float(phase_costs.get(phase, 0.0)),
            tc_cost=0.0,
            duration_ms=int(phase_tracker.phase_timings.get(phase, 0)),
            reason_codes=phase_tracker.phase_reason_codes.get(phase, []),
        )
        for phase in phase_order
    ]
    run_ledger.top_reason_codes = [
        ReasonSummary(**item) for item in summarize_reason_codes(phase_tracker.reason_events)
    ]
    run_ledger.claim_entries = list(phase_tracker.claim_entries.values())
    run_ledger.counts.llm_calls_total = len(
        [e for e in summary_obj.events if e.provider == "openai"]
    )

    ledger.set_phase_usage([pu.to_dict() for pu in run_ledger.phase_usage])
    ledger.set_reason_summaries([rs.to_dict() for rs in run_ledger.top_reason_codes])

    summary = ledger.get_summary().to_dict()
    payload["cost_summary"] = summary

    # --- Backward-compatible, UI-safe cost fields ---
    total_credits = float(summary.get("total_credits") or 0.0)
    payload["credits_used"] = total_credits
    payload["credits_used_display"] = f"{total_credits:.2f}"
    payload["credits_used_micro"] = int(round(total_credits * 10000.0))
    payload["cost"] = float(f"{total_credits:.2f}")

    audit = payload.get("audit") or {}
    audit["usage_ledger"] = run_ledger.to_dict()
    payload["audit"] = audit

    Trace.event(
        "usage_ledger.summary",
        {
            "counts": run_ledger.counts.to_dict(),
            "phase_usage": [
                {
                    "phase": pu.phase,
                    "sc_cost": pu.sc_cost,
                    "tc_cost": pu.tc_cost,
                    "duration_ms": pu.duration_ms,
                }
                for pu in run_ledger.phase_usage
            ],
            "top_reason_codes": [
                rc.to_dict() for rc in run_ledger.top_reason_codes
            ],
        },
    )
    Trace.event("cost_summary.attached", {
        "total_credits": summary.get("total_credits"),
        "total_usd": summary.get("total_usd"),
        "event_count": len(summary.get("events", [])),
    })
    return payload
