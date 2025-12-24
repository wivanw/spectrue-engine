# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.billing.cost_event import CostEvent
from spectrue_core.billing.cost_ledger import CostLedger


def test_cost_ledger_add_get_events() -> None:
    ledger = CostLedger(run_id="run-1")
    event = CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=5)
    ledger.record_event(event)
    assert ledger.events == [event]


def test_summary_aggregation_by_stage() -> None:
    ledger = CostLedger(run_id="run-2")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=5))
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.02, cost_credits=2))
    ledger.record_event(CostEvent(stage="extract", provider="tavily", cost_usd=0.01, cost_credits=1))
    summary = ledger.get_summary()
    assert summary.by_stage_usd["search"] == 0.07
    assert summary.by_stage_credits["search"] == 7
    assert summary.by_stage_usd["extract"] == 0.01
    assert summary.by_stage_credits["extract"] == 1


def test_summary_aggregation_by_provider() -> None:
    ledger = CostLedger(run_id="run-3")
    ledger.record_event(CostEvent(stage="search", provider="tavily", cost_usd=0.05, cost_credits=5))
    ledger.record_event(CostEvent(stage="clean", provider="openai", cost_usd=0.02, cost_credits=2))
    ledger.record_event(CostEvent(stage="score", provider="openai", cost_usd=0.01, cost_credits=1))
    summary = ledger.get_summary()
    assert summary.by_provider_usd["tavily"] == 0.05
    assert summary.by_provider_credits["tavily"] == 5
    assert summary.by_provider_usd["openai"] == 0.03
    assert summary.by_provider_credits["openai"] == 3


def test_zero_cost_events_are_logged() -> None:
    ledger = CostLedger(run_id="run-4")
    ledger.record_event(CostEvent(stage="cache", provider="tavily", cost_usd=0.0, cost_credits=0))
    summary = ledger.get_summary()
    assert summary.total_usd == 0.0
    assert summary.total_credits == 0
    assert len(summary.events) == 1
