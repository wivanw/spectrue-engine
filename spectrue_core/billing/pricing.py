# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Pricing calculations for Tavily usage and LLM token costs.

All credit calculations return Decimal for fractional SC precision.
Rounding (ceiling) is applied only at the SaaS billing layer, not here.
"""

from __future__ import annotations

import math
from decimal import Decimal

from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice, RoundingStrategy


def apply_rounding(value: float, strategy: RoundingStrategy) -> int:
    """Apply rounding strategy to a float value.
    
    NOTE: This function is kept for backward compatibility but should NOT be used
    for credit calculations. The SaaS layer applies ceiling rounding at final charge.
    """
    if strategy == "ceil":
        return int(math.ceil(value))
    if strategy == "round_up_to_5":
        return int(math.ceil(value / 5.0) * 5)
    return int(round(value))


def tavily_usd_to_credits(usd_cost: float, policy: CreditPricingPolicy) -> Decimal:
    """Convert Tavily USD cost to SC as Decimal.
    
    No rounding applied - caller (SaaS layer) decides rounding strategy.
    """
    return Decimal(str(usd_cost)) / Decimal(str(policy.usd_per_spectrue_credit))


def llm_usage_to_credits(
    *,
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None,
    policy: CreditPricingPolicy,
) -> Decimal:
    """Convert LLM token usage to SC as Decimal.
    
    No rounding applied - caller (SaaS layer) decides rounding strategy.
    """
    reasoning_cost = 0.0
    if reasoning_tokens and price.usd_per_reasoning_token is not None:
        reasoning_cost = reasoning_tokens * price.usd_per_reasoning_token
    usd = (
        input_tokens * price.usd_per_input_token
        + output_tokens * price.usd_per_output_token
        + reasoning_cost
    )
    usd *= policy.llm_safety_multiplier
    return Decimal(str(usd)) / Decimal(str(policy.usd_per_spectrue_credit))


class TavilyPriceCalculator:
    """Calculate Tavily costs in USD and SC (Decimal)."""
    
    def __init__(self, policy: CreditPricingPolicy) -> None:
        self._policy = policy

    def usd_cost(self, tavily_credits_used: float) -> float:
        return (
            tavily_credits_used
            * self._policy.tavily_usd_per_credit
            * self._policy.tavily_usd_multiplier
        )

    def credits_cost(self, tavily_credits_used: float) -> Decimal:
        """Return SC cost as Decimal. No rounding."""
        return tavily_usd_to_credits(self.usd_cost(tavily_credits_used), self._policy)


class LLMPriceCalculator:
    """Calculate LLM costs in USD and SC (Decimal)."""
    
    def __init__(self, policy: CreditPricingPolicy) -> None:
        self._policy = policy

    def usd_cost(
        self,
        *,
        price: ModelPrice,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None = None,
    ) -> float:
        reasoning_cost = 0.0
        if reasoning_tokens and price.usd_per_reasoning_token is not None:
            reasoning_cost = reasoning_tokens * price.usd_per_reasoning_token
        usd = (
            input_tokens * price.usd_per_input_token
            + output_tokens * price.usd_per_output_token
            + reasoning_cost
        )
        return usd * self._policy.llm_safety_multiplier

    def credits_cost(
        self,
        *,
        price: ModelPrice,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None = None,
    ) -> Decimal:
        """Return SC cost as Decimal. No rounding."""
        usd = self.usd_cost(
            price=price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        return Decimal(str(usd)) / Decimal(str(self._policy.usd_per_spectrue_credit))

