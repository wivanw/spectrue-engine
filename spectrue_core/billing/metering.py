# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Metering helpers for Tavily and LLM usage."""

from __future__ import annotations

import logging
from typing import Any
from decimal import Decimal

from spectrue_core.billing.cost_event import CostEvent
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.pricing import LLMPriceCalculator, TavilyPriceCalculator
from spectrue_core.billing.token_estimator import estimate_completion_usage
from spectrue_core.billing.types import CreditPricingPolicy, ModelPrice

logger = logging.getLogger(__name__)


def _extract_tavily_credits(response: dict[str, Any] | None) -> float:
    """Extract Tavily credits used from response.
    
    Note: Tavily API doesn't always return credits in the response body.
    Returns FALLBACK_CREDITS_PER_CALL (1.0) if credits not found.
    """
    FALLBACK_CREDITS_PER_CALL = 1.0  # Tavily charges 1 credit per search/extract
    
    if not response:
        return FALLBACK_CREDITS_PER_CALL  # Assume 1 credit was used
    for key in ("credits", "credits_used", "creditsUsed", "total_credits"):
        if key in response:
            try:
                return float(response[key])
            except (TypeError, ValueError):
                return FALLBACK_CREDITS_PER_CALL
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("credits", "credits_used", "total_credits"):
            if key in usage:
                try:
                    return float(usage[key])
                except (TypeError, ValueError):
                    return FALLBACK_CREDITS_PER_CALL
    return FALLBACK_CREDITS_PER_CALL  # Fallback: 1 Tavily credit per call


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

    def record_embedding(
        self,
        *,
        model: str,
        stage: str | None,
        usage: dict[str, Any] | None,
        input_texts: list[str],
        meta: dict[str, Any] | None = None,
    ) -> CostEvent:
        """Record cost for an embeddings API call.

        Embeddings are treated as input-only tokens. If token usage is missing,
        we fall back to a rough text-length token estimate.

        Fail-soft: if pricing for the embedding model is missing in the policy,
        record a zero-cost event instead of raising.
        """
        try:
            price = self._get_model_price(model)
        except Exception as e:
            event = CostEvent(
                stage=stage or "embed",
                provider=self._provider,
                cost_usd=0.0,
                cost_credits=Decimal("0"),
                run_id=self._ledger.run_id,
                meta={
                    **(meta or {}),
                    "model": model,
                    "missing_price": True,
                    "error": str(e)[:120],
                },
            )
            self._ledger.record_event(event)
            return event

        tokens = None
        if isinstance(usage, dict):
            # OpenAI embeddings typically returns total_tokens
            tokens = usage.get("total_tokens") or usage.get("input_tokens")

        if tokens is None:
            approx = 0
            for t in input_texts or []:
                approx += max(1, len(t or "") // 4)
            tokens = approx

        usd_cost = self._calculator.usd_cost(price=price, input_tokens=int(tokens), output_tokens=0)
        credits_cost = self._calculator.credits_cost(price=price, input_tokens=int(tokens), output_tokens=0)

        event = CostEvent(
            stage=stage or "embed",
            provider=self._provider,
            cost_usd=usd_cost,
            cost_credits=credits_cost,
            run_id=self._ledger.run_id,
            meta=meta
            or {
                "model": model,
                "input_tokens": int(tokens),
                "batch_size": len(input_texts or []),
            },
        )
        self._ledger.record_event(event)
        return event

