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

from spectrue_core.billing.cost_event import CostEvent
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.progress_emitter import CostProgressEmitter


def test_emit_delta_when_threshold_exceeded() -> None:
    ledger = CostLedger(run_id="run-1")
    emitter = CostProgressEmitter(ledger=ledger, min_delta_to_show=5, emit_cost_deltas=True)
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=3))
    assert emitter.maybe_emit(stage="searching") is None
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=3))
    payload = emitter.maybe_emit(stage="searching")
    assert payload is not None
    assert payload.delta == 6
    assert payload.total == 6


def test_no_emission_below_threshold() -> None:
    ledger = CostLedger(run_id="run-2")
    emitter = CostProgressEmitter(ledger=ledger, min_delta_to_show=5, emit_cost_deltas=True)
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.01, cost_credits=2))
    payload = emitter.maybe_emit(stage="searching")
    assert payload is None
