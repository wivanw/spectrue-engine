# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Pricing calculations for Tavily usage and LLM token costs."""

from __future__ import annotations

import math

from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice, RoundingStrategy


def apply_rounding(value: float, strategy: RoundingStrategy) -> int:
    if strategy == "ceil":
        return int(math.ceil(value))
    if strategy == "round_up_to_5":
        return int(math.ceil(value / 5.0) * 5)
    return int(round(value))


def tavily_usd_to_credits(usd_cost: float, policy: CreditPricingPolicy) -> int:
    credits = usd_cost / policy.usd_per_spectrue_credit
    return apply_rounding(credits, policy.rounding)


def llm_usage_to_credits(
    *,
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None,
    policy: CreditPricingPolicy,
) -> int:
    reasoning_cost = 0.0
    if reasoning_tokens and price.usd_per_reasoning_token is not None:
        reasoning_cost = reasoning_tokens * price.usd_per_reasoning_token
    usd = (
        input_tokens * price.usd_per_input_token
        + output_tokens * price.usd_per_output_token
        + reasoning_cost
    )
    usd *= policy.llm_safety_multiplier
    credits = usd / policy.usd_per_spectrue_credit
    return apply_rounding(credits, policy.rounding)


class TavilyPriceCalculator:
    def __init__(self, policy: CreditPricingPolicy) -> None:
        self._policy = policy

    def usd_cost(self, tavily_credits_used: float) -> float:
        return (
            tavily_credits_used
            * self._policy.tavily_usd_per_credit
            * self._policy.tavily_usd_multiplier
        )

    def credits_cost(self, tavily_credits_used: float) -> int:
        return tavily_usd_to_credits(self.usd_cost(tavily_credits_used), self._policy)


class LLMPriceCalculator:
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
    ) -> int:
        usd = self.usd_cost(
            price=price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        credits = usd / self._policy.usd_per_spectrue_credit
        return apply_rounding(credits, self._policy.rounding)

