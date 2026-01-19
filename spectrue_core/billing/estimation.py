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
"""Cost estimation utilities for pre-run credit ranges."""

from __future__ import annotations

from dataclasses import dataclass

from spectrue_core.billing.pricing import LLMPriceCalculator, TavilyPriceCalculator
from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice


@dataclass(frozen=True, slots=True)
class StageTokenEstimate:
    input_tokens: int
    output_tokens: int
    per_claim_input: int = 0
    per_claim_output: int = 0


# Calibrated based on real pipeline traces (2025-12-26)
# Previous estimates were ~3x higher than actual usage
DEFAULT_STAGE_ESTIMATES: dict[str, StageTokenEstimate] = {
    "extract": StageTokenEstimate(input_tokens=400, output_tokens=200),  # URL extraction (if applicable)
    "clean": StageTokenEstimate(input_tokens=600, output_tokens=300),    # Article cleaning (if applicable)
    "claims": StageTokenEstimate(
        input_tokens=500, output_tokens=400, per_claim_input=50, per_claim_output=30
    ),  # Claim extraction: reduced based on actual traces (was 2500/800)
    "clustering": StageTokenEstimate(
        input_tokens=200, output_tokens=50, per_claim_input=50, per_claim_output=25
    ),  # Stance clustering: minimal for simple checks
    "scoring": StageTokenEstimate(
        input_tokens=1000, output_tokens=200, per_claim_input=1000, per_claim_output=150
    ),  # Score evidence: heavy context reading (trace showed ~3k tokens for 2 claims)
    "claim_retrieval_plan": StageTokenEstimate(
        input_tokens=400, output_tokens=400, per_claim_input=50, per_claim_output=50
    ),  # Planning retrival strategies per claim
    "edge_typing": StageTokenEstimate(
        input_tokens=500, output_tokens=200, per_claim_input=50, per_claim_output=20
    ),  # Claim graph edge classification
}


class CostEstimator:
    def __init__(
        self,
        policy: CreditPricingPolicy,
        *,
        standard_model: str,
        pro_model: str,
        standard_search_credits: int = 1,
        pro_search_credits: int = 2,
        stage_estimates: dict[str, StageTokenEstimate] | None = None,
    ) -> None:
        self._policy = policy
        self._standard_model = standard_model
        self._pro_model = pro_model
        self._standard_search_credits = standard_search_credits
        self._pro_search_credits = pro_search_credits
        self._stage_estimates = stage_estimates or DEFAULT_STAGE_ESTIMATES
        self._llm_calc = LLMPriceCalculator(policy)
        self._tavily_calc = TavilyPriceCalculator(policy)

    def _get_model_price(self, model: str) -> ModelPrice:
        price = self._policy.get_model_price(model)
        if price is None:
            raise ValueError(f"Missing pricing for model: {model}")
        return price

    def _estimate_llm_stage(self, *, model: str, stage: str, claim_count: int, input_length: int) -> int:
        estimate = self._stage_estimates.get(stage)
        if not estimate:
            return 0
        
        # Dynamic base input tokens based on input length (approx 1 token per 3-4 chars)
        # We assume instructions take ~500-1000 tokens overhead.
        # Stages reading the full text: extract, clean, claims.
        # Stages reading snippets: clustering, scoring.
        
        base_input = estimate.input_tokens
        if stage in ("extract", "clean", "claims"):
             # Heuristic: base represents instructions overhead, add text length
             text_tokens = int(input_length / 3.5)
             base_input = estimate.input_tokens + text_tokens
        
        input_tokens = base_input + estimate.per_claim_input * max(0, claim_count)
        output_tokens = estimate.output_tokens + estimate.per_claim_output * max(0, claim_count)
        
        price = self._get_model_price(model)
        return self._llm_calc.credits_cost(
            price=price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=None,
        )

    def estimate_min(
        self, *, claim_count: int, search_count: int, input_type: str = "text", input_length: int = 1000
    ) -> int:
        return self.estimate_range(
            claim_count=claim_count, search_count=search_count, input_type=input_type, input_length=input_length
        )["min"]

    def estimate_max(
        self, *, claim_count: int, search_count: int, input_type: str = "text", input_length: int = 1000
    ) -> int:
        return self.estimate_range(
            claim_count=claim_count, search_count=search_count, input_type=input_type, input_length=input_length
        )["max"]

    def estimate_range(
        self,
        *,
        claim_count: int,
        search_count: int,
        input_type: str = "text",
        input_length: int = 1000,
    ) -> dict:
        min_by_stage: dict[str, int] = {}
        max_by_stage: dict[str, int] = {}

        # If input is just text/query, we skip extraction and cleaning stages
        stages_to_run = list(self._stage_estimates.keys())
        if input_type == "text":
            if "extract" in stages_to_run:
                stages_to_run.remove("extract")
            if "clean" in stages_to_run:
                stages_to_run.remove("clean")

        for stage in stages_to_run:
            min_by_stage[stage] = self._estimate_llm_stage(
                model=self._standard_model,
                stage=stage,
                claim_count=claim_count,
                input_length=input_length,
            )
            max_by_stage[stage] = self._estimate_llm_stage(
                model=self._pro_model,
                stage=stage,
                claim_count=claim_count,
                input_length=input_length,
            )

        search_usd_min = self._tavily_calc.usd_cost(search_count * self._standard_search_credits)
        search_usd_max = self._tavily_calc.usd_cost(search_count * self._pro_search_credits)
        min_by_stage["search"] = self._tavily_calc.credits_cost(search_count * self._standard_search_credits)
        max_by_stage["search"] = self._tavily_calc.credits_cost(search_count * self._pro_search_credits)

        estimate_min = sum(min_by_stage.values())
        estimate_max = sum(max_by_stage.values())

        return {
            "min": estimate_min,
            "max": max(estimate_min, estimate_max),
            "by_stage": {
                stage: {"min": min_by_stage[stage], "max": max_by_stage[stage]}
                for stage in min_by_stage
            },
            "meta": {
                "search_usd_min": search_usd_min,
                "search_usd_max": search_usd_max,
            },
        }
