# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
LLM Router for multi-endpoint support.

Routes LLM calls to either OpenAI API or local OpenAI-compatible endpoints
(vLLM, TGI, etc.) based on model configuration.

The routing decision is based on a configured set of local model names,
NOT on model name prefixes.
"""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.agents.llm_client import LLMClient, ReasoningEffort
from spectrue_core.llm.model_registry import ModelID

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Routes LLM calls to appropriate client based on model configuration.
    
    - Models in `chat_model_names` → chat_client (Chat Completions API)
    - All other models → openai_client (Responses API)
    
    Example:
        openai_client = LLMClient(openai_api_key="sk-...")
        chat_client = LLMClient(base_url="https://api.deepseek.com")
        router = LLMRouter(
            openai_client=openai_client,
            chat_client=chat_client,
            chat_model_names=[ModelID.MID, "deepseek-reasoner"],
        )
        
        # Routes to local_client
        result = await router.call_json(model="qwen/qwen3-14b", ...)
        
        # Routes to openai_client
        result = await router.call_json(model=ModelID.NANO, ...)
    """

    def __init__(
        self,
        *,
        openai_client: LLMClient,
        chat_client: LLMClient | None = None,
        chat_model_names: list[str] | None = None,
    ):
        """
        Initialize LLM Router.
        
        Args:
            openai_client: Client for OpenAI API (Responses API)
            chat_client: Client for Chat Completions API (DeepSeek, etc.).
                          If None, all calls go to openai_client.
            chat_model_names: List of model names that should use chat_client.
                               Matching is case-insensitive.
        """
        self.openai_client = openai_client
        self.chat_client = chat_client
        # Normalize to lowercase for case-insensitive matching
        self._chat_model_names: set[str] = set()
        if chat_model_names:
            self._chat_model_names = {name.lower() for name in chat_model_names}
        
        logger.debug(
            "[LLMRouter] Initialized with chat_models=%s, chat_client=%s",
            self._chat_model_names or "none",
            "configured" if chat_client else "none",
        )

    def _get_client(self, model: str) -> LLMClient:
        """
        Determine which client to use for the given model.
        
        Args:
            model: Model name to route
            
        Returns:
            Appropriate LLMClient instance
        """
        if self.chat_client and model.lower() in self._chat_model_names:
            logger.debug("[LLMRouter] Routing model '%s' to chat_client", model)
            return self.chat_client
        logger.debug("[LLMRouter] Routing model '%s' to openai_client", model)
        return self.openai_client

    async def call(
        self,
        *,
        model: str,
        input: str,  # noqa: A002
        instructions: str | None = None,
        json_output: bool = False,
        response_schema: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort = "low",
        cache_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        stage: str | None = None,
    ) -> dict:
        """
        Route and execute LLM call.
        
        Passes all parameters to the selected client. For local endpoints,
        Responses-specific parameters (reasoning_effort, cache_key) are
        ignored by the client internally.
        """
        client = self._get_client(model)
        return await client.call(
            model=model,
            input=input,
            instructions=instructions,
            json_output=json_output,
            response_schema=response_schema,
            reasoning_effort=reasoning_effort,
            cache_key=cache_key,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
            stage=stage,
        )

    async def call_json(
        self,
        *,
        model: str,
        input: str,  # noqa: A002
        instructions: str | None = None,
        response_schema: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort = "low",
        cache_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
    ) -> dict:
        """
        Route and execute LLM call, returning parsed JSON directly.
        
        This method mirrors LLMClient.call_json() interface.
        """
        client = self._get_client(model)
        return await client.call_json(
            model=model,
            input=input,
            instructions=instructions,
            response_schema=response_schema,
            reasoning_effort=reasoning_effort,
            cache_key=cache_key,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
        )

    async def call_structured(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        schema: dict[str, Any],
        schema_name: str = "structured_output",
        model: str = ModelID.NANO,
        reasoning_effort: ReasoningEffort = "low",
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        temperature: float | None = None,
    ) -> dict:
        """
        Route and execute structured output call.
        
        This method mirrors LLMClient.call_structured() interface.
        """
        client = self._get_client(model)
        return await client.call_structured(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema,
            schema_name=schema_name,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
            temperature=temperature,
        )

    @property
    def _meter(self) -> Any | None:
        """Get the current meter."""
        return getattr(self.openai_client, "_meter", None)

    @_meter.setter
    def _meter(self, meter: Any | None) -> None:
        """Set the meter for both underlying clients."""
        if self.openai_client:
            self.openai_client._meter = meter
        if self.chat_client:
            self.chat_client._meter = meter

    async def close(self) -> None:
        """Clean up resources for both clients."""
        if self.openai_client:
            await self.openai_client.close()
        if self.chat_client:
            await self.chat_client.close()

