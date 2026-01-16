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
"""Immutable cost event records and summary aggregation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class CostEvent:
    """Immutable record of a single cost event.
    
    cost_credits is stored as Decimal for fractional SC precision.
    No intermediate rounding - ceiling is applied only at final charge.
    """

    stage: str
    provider: str
    cost_usd: float
    cost_credits: Decimal  # Fractional SC, no rounding during accumulation
    run_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "cost_usd": float(self.cost_usd),
            "cost_credits": round(float(self.cost_credits), 6),  # Round to avoid FP artifacts
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class RunCostSummary:
    """Aggregated totals and breakdowns for a run.
    
    All credit values are Decimal for fractional SC precision.
    Final charge should apply ceil() on total_credits.
    """

    run_id: str | None
    total_usd: float
    total_credits: Decimal  # Fractional SC total
    by_stage_usd: dict[str, float] = field(default_factory=dict)
    by_stage_credits: dict[str, Decimal] = field(default_factory=dict)
    by_provider_usd: dict[str, float] = field(default_factory=dict)
    by_provider_credits: dict[str, Decimal] = field(default_factory=dict)
    events: list[CostEvent] = field(default_factory=list)
    phase_usage: list[dict[str, Any]] = field(default_factory=list)
    reason_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_usd": float(self.total_usd),
            "total_credits": round(float(self.total_credits), 6),  # Round to avoid FP artifacts
            "by_stage_usd": dict(self.by_stage_usd),
            "by_stage_credits": {k: round(float(v), 6) for k, v in self.by_stage_credits.items()},
            "by_provider_usd": dict(self.by_provider_usd),
            "by_provider_credits": {k: round(float(v), 6) for k, v in self.by_provider_credits.items()},
            "events": [event.to_dict() for event in self.events],
            "phase_usage": list(self.phase_usage),
            "reason_summaries": list(self.reason_summaries),
        }
