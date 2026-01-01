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
from typing import Any, Literal

from openai import AsyncOpenAI

from spectrue_core.billing.metering import LLMMeter
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


ReasoningEffort = Literal["low", "medium", "high"]
CacheRetention = Literal["in_memory", "24h"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# NOTE: Minimal sanity-checker for structured outputs (not full JSON Schema).
def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
    errors: list[str] | None = None,
    max_errors: int = 5,
) -> list[str]:
    if errors is None:
        errors = []

    if len(errors) >= max_errors:
        return errors

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        for option in any_of:
            option_errors: list[str] = []
            _validate_schema(value, option, path=path, errors=option_errors, max_errors=max_errors)
            if not option_errors:
                return errors
        errors.append(f"{path}: does not match anyOf schemas")
        return errors

    if "enum" in schema:
        allowed = schema.get("enum", [])
        if value not in allowed:
            errors.append(f"{path}: value {value!r} not in enum")
            return errors

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return errors
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: missing required field")
                    if len(errors) >= max_errors:
                        return errors
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], path=f"{path}.{key}", errors=errors, max_errors=max_errors)
            else:
                if isinstance(additional, dict):
                    _validate_schema(item, additional, path=f"{path}.{key}", errors=errors, max_errors=max_errors)
                elif additional is False:
                    value_repr = repr(item)
                    if len(value_repr) > 160:
                        value_repr = value_repr[:157] + "..."
                    errors.append(f"{path}.{key}: unexpected field {key!r}={value_repr}")
            if len(errors) >= max_errors:
                return errors
        return errors

    if schema_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array")
            return errors
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path}: expected at most {max_items} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema(
                    item,
                    items_schema,
                    path=f"{path}[{idx}]",
                    errors=errors,
                    max_errors=max_errors,
                )
                if len(errors) >= max_errors:
                    return errors
        return errors

    if schema_type == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
            return errors
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(f"{path}: expected minLength {min_len}")
        if isinstance(max_len, int) and len(value) > max_len:
            errors.append(f"{path}: expected maxLength {max_len}")
        return errors

    if schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer")
            return errors
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: expected >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: expected <= {maximum}")
        return errors

    if schema_type == "number":
        if not _is_number(value):
            errors.append(f"{path}: expected number")
            return errors
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: expected >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: expected <= {maximum}")
        return errors

    if schema_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
        return errors

    return errors


_SCHEMA_ERROR_MARKERS = (
    "llm schema validation failed",
    "invalid_json_schema",
    "text.format.schema",
    "text.format.name",
    "missing_explainability_score",
    "invalid_explainability_score",
)


def is_schema_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _SCHEMA_ERROR_MARKERS)


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
        default_timeout: float = 60.0,  # Increased from 30.0 for complex tasks
        max_retries: int = 3,
        cache_retention: CacheRetention = "in_memory", # Fix default
        meter: LLMMeter | None = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            default_timeout: Default timeout for API calls in seconds
            max_retries: Maximum retry attempts on failure
            cache_retention: Prompt cache retention ("in_memory" or "24h")
        """
        # Disable internal retries so we control them explicitly
        self.client = AsyncOpenAI(api_key=openai_api_key, max_retries=0)
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.cache_retention = cache_retention
        self._sem = asyncio.Semaphore(8)  # Concurrency limit
        self._meter = meter

    async def call(
        self,
        *,
        model: str,
        input: str,  # noqa: A002 - 'input' is the official API param name
        instructions: str | None = None,
        json_output: bool = False,
        response_schema: dict[str, Any] | None = None,
        reasoning_effort: ReasoningEffort = "low",
        cache_key: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        stage: str | None = None,
    ) -> dict:
        """
        Execute LLM call using Responses API.
        
        Args:
            model: Model to use (e.g., "gpt-5-nano", "gpt-5-mini")
            input: The main input/prompt content
            instructions: System instructions (cached if cache_key provided)
            json_output: If True, request JSON output and parse response
            response_schema: Optional JSON schema for strict structured output
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

        json_output_requested = json_output or response_schema is not None

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
        if json_output_requested:
            if response_schema:
                schema_name = response_schema.get("title") if isinstance(response_schema, dict) else None
                if not schema_name:
                    schema_name = "structured_output"
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
                    }
                }
            else:
                params["text"] = {"format": {"type": "json_object"}}

        # Reasoning control for GPT-5 and O-series models
        if "gpt-5" in model or model.startswith("o"):
            params["reasoning"] = {"effort": reasoning_effort}

        # Prompt caching
        if cache_key:
            params["prompt_cache_key"] = cache_key
            # Fix retention literal (in_memory vs in-memory)
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
            "json_output": json_output_requested,
            "response_schema": bool(response_schema),
            "cache_key": cache_key,
            # Log full prompts for debugging
            "input_text": input[:10000], # Truncate to 10k chars
            "instructions_text": (instructions or "")[:5000],
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
                if json_output_requested:
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning("[LLMClient] JSON parse failed: %s", e)
                        # Log detailed parse error with context
                        error_context = content[max(0, e.pos - 50):e.pos + 50] if e.pos else content[:200]
                        Trace.event(
                            f"{trace_kind}.json_parse_error",
                            {
                                "model": model,
                                "error": str(e),
                                "error_position": e.pos,
                                "error_context": error_context,
                                "content_length": len(content),
                                "content_head": content[:500],
                                "payload_hash": payload_hash,
                            },
                        )
                        # Try to extract JSON from markdown code block
                        if "```json" in content:
                            try:
                                json_block = content.split("```json")[1].split("```")[0]
                                parsed = json.loads(json_block.strip())
                                Trace.event(
                                    f"{trace_kind}.json_recovered_from_markdown",
                                    {"model": model, "payload_hash": payload_hash},
                                )
                            except (IndexError, json.JSONDecodeError) as md_e:
                                Trace.event(
                                    f"{trace_kind}.json_markdown_recovery_failed",
                                    {"model": model, "error": str(md_e), "payload_hash": payload_hash},
                                )
                        if parsed is None:
                            raise ValueError(f"LLM JSON parse failed: {e}") from e

                    if response_schema:
                        schema_errors = _validate_schema(parsed, response_schema)
                        if schema_errors:
                            # Log detailed schema validation error
                            logger.warning(
                                "[LLMClient] Schema validation failed: %s errors",
                                len(schema_errors),
                            )
                            Trace.event(
                                f"{trace_kind}.schema_validation_error",
                                {
                                    "model": model,
                                    "errors": schema_errors[:10],  # Limit to 10 errors
                                    "error_count": len(schema_errors),
                                    "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                                    "content_head": content[:1000],
                                    "payload_hash": payload_hash,
                                },
                            )
                            raise ValueError(f"LLM schema validation failed: {schema_errors[0]}")

                # Extract usage info
                usage = None
                cache_status = "NONE" # Default if no cache_key
                request_id = getattr(response, "id", "unknown")

                if response.usage:
                    # Extract detailed cache hits via prompt_tokens_details
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
                        # Log raw usage for debugging SDK mapping issues
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

                if self._meter:
                    try:
                        event = self._meter.record_completion(
                            model=response.model,
                            stage=stage or trace_kind,
                            usage=usage,
                            input_text=input,
                            output_text=content,
                            instructions=instructions,
                        )
                        Trace.event("llm.metering.recorded", {
                            "stage": stage or trace_kind,
                            "cost_credits": str(event.cost_credits),
                        })
                    except Exception as exc:
                        logger.warning("[LLMClient] Metering failed: %s", exc)
                        Trace.event("llm.metering.failed", {"error": str(exc)[:200]})
                else:
                    Trace.event("llm.metering.skipped", {"reason": "no_meter"})

                Trace.event(f"{trace_kind}.response", {
                    "model": response.model,
                    "content_chars": len(content),
                    "cache_status": cache_status,
                    "attempt": attempt + 1,
                    "payload_hash": payload_hash,
                    # Log full response for debugging
                    "response_text": content[:15000], # Truncate to 15k chars
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
        response_schema: dict[str, Any] | None = None,
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
            response_schema=response_schema,
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
