# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Immutable cost event records and summary aggregation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class CostEvent:
    """Immutable record of a single cost event."""

    stage: str
    provider: str
    cost_usd: float
    cost_credits: int
    run_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "cost_usd": float(self.cost_usd),
            "cost_credits": int(self.cost_credits),
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class RunCostSummary:
    """Aggregated totals and breakdowns for a run."""

    run_id: str | None
    total_usd: float
    total_credits: int
    by_stage_usd: dict[str, float] = field(default_factory=dict)
    by_stage_credits: dict[str, int] = field(default_factory=dict)
    by_provider_usd: dict[str, float] = field(default_factory=dict)
    by_provider_credits: dict[str, int] = field(default_factory=dict)
    events: list[CostEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_usd": float(self.total_usd),
            "total_credits": int(self.total_credits),
            "by_stage_usd": dict(self.by_stage_usd),
            "by_stage_credits": dict(self.by_stage_credits),
            "by_provider_usd": dict(self.by_provider_usd),
            "by_provider_credits": dict(self.by_provider_credits),
            "events": [event.to_dict() for event in self.events],
        }
