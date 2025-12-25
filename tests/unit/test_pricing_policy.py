# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from decimal import Decimal

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
    """TavilyPriceCalculator returns Decimal (no rounding in engine)."""
    policy = _policy()
    calc = TavilyPriceCalculator(policy)
    # 1 TC * 0.005 USD/TC * 10.0 multiplier = 0.05 USD
    # 0.05 USD / 0.01 USD/SC = 5 SC
    result = calc.credits_cost(1)
    assert isinstance(result, Decimal)
    assert result == Decimal("5")


def test_usd_to_credits_conversion() -> None:
    """tavily_usd_to_credits returns Decimal without rounding.
    
    Rounding is now caller's responsibility (SaaS layer).
    """
    policy = _policy()
    # 0.011 USD / 0.01 USD/SC = 1.1 SC (Decimal, no rounding)
    result = tavily_usd_to_credits(0.011, policy)
    assert isinstance(result, Decimal)
    assert result == Decimal("1.1")


def test_rounding_strategies() -> None:
    """apply_rounding is kept for backward compatibility but not used in engine."""
    assert apply_rounding(1.1, "ceil") == 2
    assert apply_rounding(11.1, "round_up_to_5") == 15
    assert apply_rounding(2.5, "bankers") == 2
