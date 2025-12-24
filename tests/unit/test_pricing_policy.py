# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.billing.pricing import (
    TavilyPriceCalculator,
    apply_rounding,
    tavily_usd_to_credits,
)
from spectrue_core.billing.types import CreditPricingPolicy


def _policy(rounding: str = "ceil") -> CreditPricingPolicy:
    return CreditPricingPolicy(
        usd_per_spectrue_credit=0.01,
        tavily_usd_per_credit=0.005,
        tavily_usd_multiplier=10.0,
        llm_safety_multiplier=1.2,
        rounding=rounding,
        llm_prices={},
    )


def test_tavily_multiplier_conversion() -> None:
    policy = _policy()
    calc = TavilyPriceCalculator(policy)
    assert calc.credits_cost(1) == 5


def test_usd_to_credits_conversion() -> None:
    policy = _policy()
    assert tavily_usd_to_credits(0.011, policy) == 2


def test_rounding_strategies() -> None:
    assert apply_rounding(1.1, "ceil") == 2
    assert apply_rounding(11.1, "round_up_to_5") == 15
    assert apply_rounding(2.5, "bankers") == 2
