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

from spectrue_core.billing.estimation import CostEstimator
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
                usd_per_reasoning_token=None,
            ),
            "gpt-5": ModelPrice(
                usd_per_input_token=0.00000125,
                usd_per_output_token=0.00001,
                usd_per_reasoning_token=0.00001,
            ),
        },
    )


def _estimator() -> CostEstimator:
    return CostEstimator(
        _policy(),
        standard_model="gpt-5-nano",
        pro_model="gpt-5",
    )


def test_input_type_impact() -> None:
    estimator = _estimator()
    # Default (url) vs text
    url_res = estimator.estimate_range(claim_count=2, search_count=3, input_type="url")
    text_res = estimator.estimate_range(claim_count=2, search_count=3, input_type="text")
    
    # Text should be cheaper because extract/clean are skipped
    assert text_res["min"] < url_res["min"]
    assert "extract" not in text_res["by_stage"]
    assert "clean" not in text_res["by_stage"]
    assert "extract" in url_res["by_stage"]

def test_text_only_stages() -> None:
    estimator = _estimator()
    text_res = estimator.estimate_range(claim_count=1, search_count=1, input_type="text")
    stages = text_res["by_stage"].keys()
    assert "extract" not in stages
    assert "clean" not in stages
    assert "claims" in stages
    assert "clustering" in stages
    assert "scoring" in stages
