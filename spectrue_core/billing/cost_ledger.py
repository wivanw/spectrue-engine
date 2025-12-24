# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Cost ledger to track and summarize billing events per run."""

from __future__ import annotations

from dataclasses import dataclass, field

from spectrue_core.billing.cost_event import CostEvent, RunCostSummary


@dataclass(slots=True)
class CostLedger:
    run_id: str | None = None
    events: list[CostEvent] = field(default_factory=list)

    def record_event(self, event: CostEvent) -> None:
        self.events.append(event)

    def get_summary(self) -> RunCostSummary:
        by_stage_usd: dict[str, float] = {}
        by_stage_credits: dict[str, int] = {}
        by_provider_usd: dict[str, float] = {}
        by_provider_credits: dict[str, int] = {}

        total_usd = 0.0
        total_credits = 0

        for event in self.events:
            total_usd += event.cost_usd
            total_credits += event.cost_credits

            by_stage_usd[event.stage] = by_stage_usd.get(event.stage, 0.0) + event.cost_usd
            by_stage_credits[event.stage] = (
                by_stage_credits.get(event.stage, 0) + event.cost_credits
            )
            by_provider_usd[event.provider] = (
                by_provider_usd.get(event.provider, 0.0) + event.cost_usd
            )
            by_provider_credits[event.provider] = (
                by_provider_credits.get(event.provider, 0) + event.cost_credits
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
        )
