# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Emit cost deltas during progress updates."""

from __future__ import annotations

from dataclasses import dataclass

from spectrue_core.billing.cost_ledger import CostLedger


@dataclass
class CostDeltaPayload:
    stage: str
    delta: int
    total: int


class CostProgressEmitter:
    def __init__(
        self,
        *,
        ledger: CostLedger,
        min_delta_to_show: int,
        emit_cost_deltas: bool,
    ) -> None:
        self._ledger = ledger
        self._min_delta = max(0, int(min_delta_to_show))
        self._emit = bool(emit_cost_deltas)
        self._last_total = 0

    @property
    def enabled(self) -> bool:
        return self._emit

    def reset(self) -> None:
        self._last_total = 0

    def maybe_emit(self, *, stage: str) -> CostDeltaPayload | None:
        if not self._emit:
            return None
        summary = self._ledger.get_summary()
        total = int(summary.total_credits)
        delta = total - self._last_total
        if delta < self._min_delta:
            return None
        self._last_total = total
        return CostDeltaPayload(stage=stage, delta=delta, total=total)
