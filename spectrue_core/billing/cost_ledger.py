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
"""Cost ledger to track and summarize billing events per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from spectrue_core.billing.cost_event import CostEvent, RunCostSummary
from spectrue_core.utils.trace import Trace


@dataclass(slots=True)
class CostLedger:
    """Tracks cost events during a run and produces a summary.
    
    All credit values are accumulated as Decimal for fractional SC precision.
    No intermediate rounding - ceiling is applied only at final charge.
    """

    run_id: str | None = None
    events: list[CostEvent] = field(default_factory=list)
    phase_usage: list[dict] = field(default_factory=list)
    reason_summaries: list[dict] = field(default_factory=list)

    def record_event(self, event: CostEvent) -> None:
        self.events.append(event)
        Trace.event("billing.cost_event", event.to_dict())

    def set_phase_usage(self, usage: list[dict]) -> None:
        self.phase_usage = list(usage)

    def set_reason_summaries(self, summaries: list[dict]) -> None:
        self.reason_summaries = list(summaries)

    @property
    def total_credits(self):
        """Backward-compatible accessor used by legacy pipeline code."""
        try:
            return self.get_summary().total_credits
        except Exception:
            return Decimal("0")

    @property
    def total_usd(self):
        """Convenience accessor for total USD in this ledger."""
        try:
            return float(self.get_summary().total_usd)
        except Exception:
            return 0.0

    def get_summary(self) -> RunCostSummary:
        by_stage_usd: dict[str, float] = {}
        by_stage_credits: dict[str, Decimal] = {}
        by_provider_usd: dict[str, float] = {}
        by_provider_credits: dict[str, Decimal] = {}

        total_usd = 0.0
        total_credits = Decimal("0")

        for event in self.events:
            total_usd += event.cost_usd
            total_credits += event.cost_credits  # Decimal accumulation

            by_stage_usd[event.stage] = by_stage_usd.get(event.stage, 0.0) + event.cost_usd
            by_stage_credits[event.stage] = (
                by_stage_credits.get(event.stage, Decimal("0")) + event.cost_credits
            )
            by_provider_usd[event.provider] = (
                by_provider_usd.get(event.provider, 0.0) + event.cost_usd
            )
            by_provider_credits[event.provider] = (
                by_provider_credits.get(event.provider, Decimal("0")) + event.cost_credits
            )

        return RunCostSummary(
            run_id=self.run_id,
            total_usd=total_usd,
            total_credits=total_credits,
            by_stage_usd=by_stage_usd,
            by_stage_credits=by_stage_credits,
            by_provider_usd=by_provider_usd,
            by_provider_credits=by_provider_credits,
            events=list(self.events),
            phase_usage=list(self.phase_usage),
            reason_summaries=list(self.reason_summaries),
        )

    def to_summary_dict(self) -> dict:
        """Convert summary to dictionary for API response."""
        summary = self.get_summary()
        from dataclasses import asdict
        return asdict(summary)
