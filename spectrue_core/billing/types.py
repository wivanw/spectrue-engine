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
"""Pricing policy types for credits-based billing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RoundingStrategy = Literal["ceil", "round_up_to_5", "bankers"]


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD pricing per token for a specific LLM model."""

    usd_per_input_token: float
    usd_per_output_token: float
    usd_per_reasoning_token: float | None = None


@dataclass(frozen=True, slots=True)
class CreditPricingPolicy:
    """Single source of truth for credit pricing policies."""

    usd_per_spectrue_credit: float
    tavily_usd_per_credit: float
    tavily_usd_multiplier: float
    llm_safety_multiplier: float
    rounding: RoundingStrategy
    min_delta_to_show: int = 5
    emit_cost_deltas: bool = False
    llm_prices: dict[str, ModelPrice] = field(default_factory=dict)

    def get_model_price(self, model: str) -> ModelPrice | None:
        return self.llm_prices.get(model)
