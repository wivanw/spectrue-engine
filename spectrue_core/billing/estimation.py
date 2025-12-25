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


DEFAULT_STAGE_ESTIMATES: dict[str, StageTokenEstimate] = {
    "extract": StageTokenEstimate(input_tokens=800, output_tokens=400),
    "clean": StageTokenEstimate(input_tokens=1200, output_tokens=600),
    "claims": StageTokenEstimate(
        input_tokens=2500, output_tokens=600, per_claim_input=200, per_claim_output=100
    ),
    "clustering": StageTokenEstimate(
        input_tokens=800, output_tokens=300, per_claim_input=100, per_claim_output=50
    ),
    "scoring": StageTokenEstimate(
        input_tokens=800, output_tokens=400, per_claim_input=150, per_claim_output=75
    ),
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

    def _estimate_llm_stage(self, *, model: str, stage: str, claim_count: int) -> int:
        estimate = self._stage_estimates.get(stage)
        if not estimate:
            return 0
        input_tokens = estimate.input_tokens + estimate.per_claim_input * max(0, claim_count)
        output_tokens = estimate.output_tokens + estimate.per_claim_output * max(0, claim_count)
        price = self._get_model_price(model)
        return self._llm_calc.credits_cost(
            price=price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=None,
        )

    def estimate_min(
        self, *, claim_count: int, search_count: int, input_type: str = "text"
    ) -> int:
        return self.estimate_range(
            claim_count=claim_count, search_count=search_count, input_type=input_type
        )["min"]

    def estimate_max(
        self, *, claim_count: int, search_count: int, input_type: str = "text"
    ) -> int:
        return self.estimate_range(
            claim_count=claim_count, search_count=search_count, input_type=input_type
        )["max"]

    def estimate_range(
        self,
        *,
        claim_count: int,
        search_count: int,
        input_type: str = "text",
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
            )
            max_by_stage[stage] = self._estimate_llm_stage(
                model=self._pro_model,
                stage=stage,
                claim_count=claim_count,
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
