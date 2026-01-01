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

from decimal import Decimal

from spectrue_core.billing.cost_event import CostEvent
from spectrue_core.billing.cost_ledger import CostLedger


def test_cost_ledger_add_get_events() -> None:
    ledger = CostLedger(run_id="run-1")
    event = CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=Decimal("5"))
    ledger.record_event(event)
    assert ledger.events == [event]


def test_summary_aggregation_by_stage() -> None:
    ledger = CostLedger(run_id="run-2")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=Decimal("5")))
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.02, cost_credits=Decimal("2")))
    ledger.record_event(CostEvent(stage="extract", provider="tavily", cost_usd=0.01, cost_credits=Decimal("1")))
    summary = ledger.get_summary()
    assert summary.by_stage_usd["search"] == 0.07
    assert summary.by_stage_credits["search"] == Decimal("7")
    assert summary.by_stage_usd["extract"] == 0.01
    assert summary.by_stage_credits["extract"] == Decimal("1")


def test_summary_aggregation_by_provider() -> None:
    ledger = CostLedger(run_id="run-3")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=Decimal("5")))
    ledger.record_event(CostEvent(stage="clean", provider="openai", cost_usd=0.02, cost_credits=Decimal("2")))
    ledger.record_event(CostEvent(stage="score", provider="openai", cost_usd=0.01, cost_credits=Decimal("1")))
    summary = ledger.get_summary()
    assert summary.by_provider_usd["tavily"] == 0.05
    assert summary.by_provider_credits["tavily"] == Decimal("5")
    assert summary.by_provider_usd["openai"] == 0.03
    assert summary.by_provider_credits["openai"] == Decimal("3")


def test_zero_cost_events_are_logged() -> None:
    ledger = CostLedger(run_id="run-4")
    ledger.record_event(CostEvent(stage="cache", provider="tavily", cost_usd=0.0, cost_credits=Decimal("0")))
    summary = ledger.get_summary()
    assert summary.total_usd == 0.0
    assert summary.total_credits == Decimal("0")
    assert len(summary.events) == 1


def test_fractional_credits_accumulation() -> None:
    """Verify that fractional SC values are accumulated correctly without rounding."""
    ledger = CostLedger(run_id="run-5")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.01, cost_credits=Decimal("1.5")))
    ledger.record_event(CostEvent(stage="llm", provider="openai", cost_usd=0.02, cost_credits=Decimal("2.37")))
    summary = ledger.get_summary()
    # 1.5 + 2.37 = 3.87 (no rounding)
    assert summary.total_credits == Decimal("3.87")
    assert isinstance(summary.total_credits, Decimal)


def test_cost_ledger_backward_compatible_total_credits_property() -> None:
    ledger = CostLedger(run_id="run-compat")
    assert ledger.total_credits == Decimal("0")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.01, cost_credits=Decimal("1.25")))
    assert ledger.total_credits == Decimal("1.25")
