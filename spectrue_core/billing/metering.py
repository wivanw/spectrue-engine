# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Metering helpers for Tavily and LLM usage."""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.billing.cost_event import CostEvent
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.pricing import LLMPriceCalculator, TavilyPriceCalculator
from spectrue_core.billing.token_estimator import estimate_completion_usage
from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice

logger = logging.getLogger(__name__)


def _extract_tavily_credits(response: dict[str, Any] | None) -> float:
    if not response:
        return 0.0
    for key in ("credits", "credits_used", "creditsUsed", "total_credits"):
        if key in response:
            try:
                return float(response[key])
            except (TypeError, ValueError):
                return 0.0
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("credits", "credits_used", "total_credits"):
            if key in usage:
                try:
                    return float(usage[key])
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


class TavilyMeter:
    def __init__(
        self,
        *,
        ledger: CostLedger,
        policy: CreditPricingPolicy,
        provider: str = "tavily",
        stage_search: str = "search",
        stage_extract: str = "extract",
    ) -> None:
        self._ledger = ledger
        self._policy = policy
        self._provider = provider
        self._stage_search = stage_search
        self._stage_extract = stage_extract
        self._calculator = TavilyPriceCalculator(policy)

    def record_search(
        self,
        *,
        response: dict[str, Any] | None = None,
        credits_used: float | None = None,
        stage: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CostEvent:
        used = credits_used if credits_used is not None else _extract_tavily_credits(response)
        usd_cost = self._calculator.usd_cost(used)
        credits_cost = self._calculator.credits_cost(used)
        event = CostEvent(
            stage=stage or self._stage_search,
            provider=self._provider,
            cost_usd=usd_cost,
            cost_credits=credits_cost,
            run_id=self._ledger.run_id,
            meta=meta or {"credits_used": used},
        )
        self._ledger.record_event(event)
        return event

    def record_extract(
        self,
        *,
        response: dict[str, Any] | None = None,
        credits_used: float | None = None,
        stage: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CostEvent:
        used = credits_used if credits_used is not None else _extract_tavily_credits(response)
        usd_cost = self._calculator.usd_cost(used)
        credits_cost = self._calculator.credits_cost(used)
        event = CostEvent(
            stage=stage or self._stage_extract,
            provider=self._provider,
            cost_usd=usd_cost,
            cost_credits=credits_cost,
            run_id=self._ledger.run_id,
            meta=meta or {"credits_used": used},
        )
        self._ledger.record_event(event)
        return event


class LLMMeter:
    def __init__(
        self,
        *,
        ledger: CostLedger,
        policy: CreditPricingPolicy,
        provider: str = "openai",
        default_stage: str = "llm",
    ) -> None:
        self._ledger = ledger
        self._policy = policy
        self._provider = provider
        self._default_stage = default_stage
        self._calculator = LLMPriceCalculator(policy)

    def _get_model_price(self, model: str) -> ModelPrice:
        price = self._policy.get_model_price(model)
        if price is None:
            raise ValueError(f"Missing pricing for model: {model}")
        return price

    def record_completion(
        self,
        *,
        model: str,
        stage: str | None,
        usage: dict[str, Any] | None,
        input_text: str,
        output_text: str,
        instructions: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> CostEvent:
        price = self._get_model_price(model)
        input_tokens = None
        output_tokens = None
        reasoning_tokens = None

        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            reasoning_tokens = usage.get("reasoning_tokens")

        if input_tokens is None or output_tokens is None:
            estimate = estimate_completion_usage(
                input_text=input_text,
                output_text=output_text,
                instructions=instructions,
            )
            input_tokens = estimate["input_tokens"]
            output_tokens = estimate["output_tokens"]
            reasoning_tokens = None

        usd_cost = self._calculator.usd_cost(
            price=price,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
        )
        credits_cost = self._calculator.credits_cost(
            price=price,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
        )

        event = CostEvent(
            stage=stage or self._default_stage,
            provider=self._provider,
            cost_usd=usd_cost,
            cost_credits=credits_cost,
            run_id=self._ledger.run_id,
            meta=meta
            or {
                "model": model,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "reasoning_tokens": int(reasoning_tokens)
                if reasoning_tokens is not None
                else None,
            },
        )
        self._ledger.record_event(event)
        return event
