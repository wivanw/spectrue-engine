# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Unified LLM client using OpenAI Responses API (M49/T180).

This module provides a simplified interface for making LLM calls with:
- Prompt caching for repeated instructions
- Reasoning control (low/medium/high effort)
- JSON output with automatic parsing
- Retry logic with graceful degradation
- Trace event logging
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from openai import AsyncOpenAI

from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


ReasoningEffort = Literal["low", "medium", "high"]
CacheRetention = Literal["in_memory", "24h"]


class LLMClient:
    """
    Unified LLM client using OpenAI Responses API.
    
    Benefits over direct chat.completions:
    - Prompt caching via prompt_cache_key (reduces tokens for repeated prompts)
    - Clear separation of instructions vs input
    - Reasoning effort control for GPT-5 models
    - Simpler JSON output configuration
    
    Example:
        client = LLMClient(openai_api_key="sk-...")
        result = await client.call(
            model="gpt-5-nano",
            input="Analyze this claim: ...",
            instructions="You are a fact-checking assistant.",
            json_output=True,
            cache_key="claim_analysis_v1",
        )
    """
    
    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        cache_retention: CacheRetention = "in_memory", # M56: Fix default
    ):
        """
        Initialize LLM client.
        
        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            default_timeout: Default timeout for API calls in seconds
            max_retries: Maximum retry attempts on failure
            cache_retention: Prompt cache retention ("in_memory" or "24h")
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.cache_retention = cache_retention
        self._sem = asyncio.Semaphore(8)  # Concurrency limit
    
    async def call(
        self,
        *,
        model: str,
        input: str,  # noqa: A002 - 'input' is the official API param name
        instructions: str | None = None,
        json_output: bool = False,
        reasoning_effort: ReasoningEffort = "low",
        cache_key: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
    ) -> dict:
        """
        Execute LLM call using Responses API.
        
        Args:
            model: Model to use (e.g., "gpt-5-nano", "gpt-5-mini")
            input: The main input/prompt content
            instructions: System instructions (cached if cache_key provided)
            json_output: If True, request JSON output and parse response
            reasoning_effort: Reasoning effort level (low/medium/high)
            cache_key: Optional key for prompt caching (reduces tokens)
            timeout: Request timeout in seconds (uses default if not specified)
            max_output_tokens: Maximum number of tokens to generate
            trace_kind: Event kind for tracing
            
        Returns:
            Dict with keys:
            - "content": Raw text content from LLM
            - "parsed": Parsed JSON if json_output=True, else None
            - "model": Model used
            - "cached": Whether prompt was cached
            - "usage": Token usage info if available
            
        Raises:
            ValueError: If response is empty after all retries
        """
        effective_timeout = timeout or self.default_timeout
        
        # Build API params
        params: dict = {
            "model": model,
            "input": input,
            "timeout": effective_timeout,
        }
        
        if instructions:
            params["instructions"] = instructions

        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        
        # JSON output configuration
        if json_output:
            params["text"] = {"format": {"type": "json_object"}}
        
        # Reasoning control for GPT-5 and O-series models
        if "gpt-5" in model or model.startswith("o"):
            params["reasoning"] = {"effort": reasoning_effort}
        
        # Prompt caching
        if cache_key:
            params["prompt_cache_key"] = cache_key
            # M56: Fix retention literal (in_memory vs in-memory)
            # CAUTION: gpt-5-nano throws 400 "invalid_parameter" for this.
            # params["prompt_cache_retention"] = self.cache_retention
        
        # Calculate payload hash for debug correlation (M55)
        # Hash includes input + instructions to match exactly what went into the prompt
        import hashlib
        import time
        payload_str = (instructions or "") + "||" + input
        payload_hash = hashlib.md5(payload_str.encode()).hexdigest()

        # Token estimation (rough: 1.3 tokens per word, slightly higher for code/json)
        est_input_tokens = int(len(input.split()) * 1.3)
        est_instr_tokens = int(len((instructions or "").split()) * 1.3)

        Trace.event(f"{trace_kind}.prompt", {
            "model": model,
            "input_chars": len(input),
            "instructions_chars": len(instructions or ""),
            "est_input_tokens": est_input_tokens,
            "est_instr_tokens": est_instr_tokens,
            "payload_hash": payload_hash,
            "json_output": json_output,
            "cache_key": cache_key,
        })
        
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                async with self._sem:
                    if attempt > 0:
                        await asyncio.sleep(0.5 * attempt)
                        logger.debug("[LLMClient] Retry %d/%d for %s", attempt + 1, self.max_retries, model)
                    
                    response = await self.client.responses.create(**params)
                
                latency_ms = int((time.time() - start_time) * 1000)

                # Extract text content
                content = response.output_text
                
                if not content or not content.strip():
                    # Check for error
                    if response.error:
                        raise ValueError(f"LLM error: {response.error}")
                    if response.status == "incomplete":
                        details = response.incomplete_details
                        raise ValueError(f"Incomplete response: {details}")
                    raise ValueError("Empty response from LLM")
                
                # Parse JSON if requested
                parsed = None
                if json_output:
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning("[LLMClient] JSON parse failed: %s", e)
                        # Try to extract JSON from markdown code block
                        if "```json" in content:
                            try:
                                json_block = content.split("```json")[1].split("```")[0]
                                parsed = json.loads(json_block.strip())
                            except (IndexError, json.JSONDecodeError):
                                pass
                        if parsed is None:
                            raise ValueError(f"Failed to parse JSON response: {e}") from e
                
                # Extract usage info
                usage = None
                cache_status = "NONE" # Default if no cache_key
                request_id = getattr(response, "id", "unknown")
                
                if response.usage:
                    # M56: Extract detailed cache hits via prompt_tokens_details
                    cached_tokens = 0
                    if hasattr(response.usage, "prompt_tokens_details"):
                        ptd = response.usage.prompt_tokens_details
                        # Support both object access and dict access depending on library version
                        if isinstance(ptd, dict):
                            cached_tokens = ptd.get("cached_tokens", 0)
                        else:
                            cached_tokens = getattr(ptd, "cached_tokens", 0)
                    
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cached_tokens": cached_tokens,
                        "latency_ms": latency_ms,
                        "request_id": request_id, 
                        # M56: Log raw usage for debugging SDK mapping issues
                        "raw": response.usage.model_dump() if hasattr(response.usage, "model_dump") else response.usage.to_dict() if hasattr(response.usage, "to_dict") else str(response.usage)
                    }

                    if cache_key:
                        if cached_tokens > 0:
                            cache_status = "HIT"
                        elif response.usage.input_tokens > 0:
                             cache_status = "MISS"
                        else:
                             cache_status = "UNKNOWN"
                    else:
                        cache_status = "NONE"

                else:
                     # Fallback if usage absent
                     cache_status = "KEY_PROVIDED" if cache_key else "NONE"
                     usage = {"latency_ms": latency_ms, "request_id": request_id}
                
                result = {
                    "content": content,
                    "parsed": parsed,
                    "model": response.model,
                    "cache_status": cache_status, 
                    "usage": usage,
                }
                
                Trace.event(f"{trace_kind}.response", {
                    "model": response.model,
                    "content_chars": len(content),
                    "cache_status": cache_status,
                    "attempt": attempt + 1,
                    "payload_hash": payload_hash
                })
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning("[LLMClient] Attempt %d failed: %s", attempt + 1, e)
                Trace.event(f"{trace_kind}.error", {
                    "model": model,
                    "attempt": attempt + 1,
                    "error": str(e)[:200],
                    "payload_hash": payload_hash
                })
        
        # All retries exhausted
        raise ValueError(f"LLM call failed after {self.max_retries} attempts: {last_error}")
    
    async def call_json(
        self,
        *,
        model: str,
        input: str,  # noqa: A002
        instructions: str | None = None,
        reasoning_effort: ReasoningEffort = "low",
        cache_key: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
    ) -> dict:
        """
        Convenience method for JSON output calls.
        
        Returns the parsed JSON directly (not wrapped in result dict).
        """
        result = await self.call(
            model=model,
            input=input,
            instructions=instructions,
            json_output=True,
            reasoning_effort=reasoning_effort,
            cache_key=cache_key,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
        )
        return result["parsed"]
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.close()
