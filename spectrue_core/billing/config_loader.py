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
"""Pricing policy configuration loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice


def load_pricing_policy(config_path: str | Path | None = None) -> CreditPricingPolicy:
    """
    Load pricing policy from JSON file with ENV overrides.

    Priority (highest to lowest):
    1. Environment variables (SPECTRUE_*)
    2. Provided config_path
    3. Default pricing JSON

    Args:
        config_path: Optional path to custom pricing config JSON

    Returns:
        CreditPricingPolicy instance
    """
    # Load base config from file
    if config_path:
        with open(config_path) as f:
            data = json.load(f)
    else:
        default_path = Path(__file__).parent / "default_pricing.json"
        if default_path.exists():
            with open(default_path) as f:
                data = json.load(f)
        else:
            data = {}

    # Apply ENV overrides
    usd_per_credit = float(
        os.environ.get(
            "SPECTRUE_USD_PER_CREDIT",
            data.get("usd_per_spectrue_credit", 0.01),
        )
    )
    tavily_per_credit = float(
        os.environ.get(
            "SPECTRUE_TAVILY_USD_PER_CREDIT",
            data.get("tavily_usd_per_credit", 0.005),
        )
    )
    tavily_multiplier = float(
        os.environ.get(
            "SPECTRUE_TAVILY_MULTIPLIER",
            data.get("tavily_usd_multiplier", 1.0),
        )
    )
    llm_safety = float(
        os.environ.get(
            "SPECTRUE_LLM_SAFETY_MULTIPLIER",
            data.get("llm_safety_multiplier", 1.2),
        )
    )
    rounding = os.environ.get(
        "SPECTRUE_ROUNDING",
        data.get("rounding", "ceil"),
    )

    # Parse LLM prices
    llm_prices_raw = data.get("llm_prices", {})
    llm_prices = {}
    for model_name, prices in llm_prices_raw.items():
        llm_prices[model_name] = ModelPrice(
            usd_per_input_token=prices.get("usd_per_input_token", 0.0),
            usd_per_output_token=prices.get("usd_per_output_token", 0.0),
            usd_per_reasoning_token=prices.get("usd_per_reasoning_token"),
        )

    return CreditPricingPolicy(
        usd_per_spectrue_credit=usd_per_credit,
        tavily_usd_per_credit=tavily_per_credit,
        tavily_usd_multiplier=tavily_multiplier,
        llm_safety_multiplier=llm_safety,
        llm_prices=llm_prices,
        rounding=rounding,
    )
