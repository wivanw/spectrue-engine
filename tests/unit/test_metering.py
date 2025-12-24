# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.metering import LLMMeter, TavilyMeter
from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice


def _policy() -> CreditPricingPolicy:
    return CreditPricingPolicy(
        usd_per_spectrue_credit=0.01,
        tavily_usd_per_credit=0.005,
        tavily_usd_multiplier=10.0,
        llm_safety_multiplier=1.2,
        rounding="ceil",
        llm_prices={
            "gpt-5-nano": ModelPrice(
                usd_per_input_token=0.00000005,
                usd_per_output_token=0.0000004,
                usd_per_reasoning_token=0.0000004,
            )
        },
    )


def test_tavily_1_credit_10x() -> None:
    ledger = CostLedger(run_id="run-1")
    meter = TavilyMeter(ledger=ledger, policy=_policy())
    meter.record_search(credits_used=1)
    assert ledger.events[0].cost_credits == 5


def test_llm_usage_tokens() -> None:
    ledger = CostLedger(run_id="run-2")
    meter = LLMMeter(ledger=ledger, policy=_policy())
    usage = {"input_tokens": 100000, "output_tokens": 200000, "reasoning_tokens": 50000}
    meter.record_completion(
        model="gpt-5-nano",
        stage="clean",
        usage=usage,
        input_text="input",
        output_text="output",
    )
    assert ledger.events[0].cost_credits == 13


def test_llm_fallback_when_usage_missing() -> None:
    ledger = CostLedger(run_id="run-3")
    meter = LLMMeter(ledger=ledger, policy=_policy())
    meter.record_completion(
        model="gpt-5-nano",
        stage="clean",
        usage=None,
        input_text="hello world",
        output_text="ok",
    )
    assert ledger.events[0].cost_credits == 1


def test_safety_multiplier_applied() -> None:
    ledger = CostLedger(run_id="run-4")
    meter = LLMMeter(ledger=ledger, policy=_policy())
    usage = {"input_tokens": 0, "output_tokens": 22500}
    meter.record_completion(
        model="gpt-5-nano",
        stage="clean",
        usage=usage,
        input_text="input",
        output_text="",
    )
    assert ledger.events[0].cost_credits == 2
