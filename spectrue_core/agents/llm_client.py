# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unified LLM client using OpenAI Responses API.

This module provides a simplified interface for making LLM calls with:
- Prompt caching for repeated instructions
- Reasoning control (low/medium/high effort)
- JSON output with automatic parsing
- Retry logic with graceful degradation
- Trace event logging
- Support for OpenAI-compatible local endpoints (vLLM, TGI) via Chat Completions API
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI

from spectrue_core.billing.metering import LLMMeter
from spectrue_core.billing.meter_context import get_current_llm_meter
from spectrue_core.utils.trace import Trace
from spectrue_core.llm.errors import LLMFailureKind
from spectrue_core.llm.errors import LLMCallError

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
        base_url: str | None = None,
        default_timeout: float = 60.0,  # Increased from 30.0 for complex tasks
        max_retries: int = 3,
        cache_retention: CacheRetention = "in_memory", # Fix default
        meter: LLMMeter | None = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Optional base URL for OpenAI-compatible endpoints (vLLM, TGI).
                      When set, uses Chat Completions API instead of Responses API.
            default_timeout: Default timeout for API calls in seconds
            max_retries: Maximum retry attempts on failure
            cache_retention: Prompt cache retention ("in_memory" or "24h")
        """
        # Disable internal retries so we control them explicitly
        client_kwargs: dict[str, Any] = {"max_retries": 0}
        if openai_api_key:
            client_kwargs["api_key"] = openai_api_key
        elif base_url:
            # Local endpoints may not require auth, but OpenAI SDK requires api_key
            # Use a dummy key for local servers
            client_kwargs["api_key"] = "local-no-key"
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.base_url = base_url  # Track if using local endpoint
        self._use_chat_completions = base_url is not None  # Local endpoints use Chat Completions API
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.cache_retention = cache_retention
        self._sem = asyncio.Semaphore(8)  # Concurrency limit
        self._meter = meter

    @staticmethod
    def _extract_first_json(raw: str) -> str | None:
        """
        Deterministically extract the first JSON object/array from raw text.
        Balanced brace/bracket scanner that respects quoted strings and escapes.
        Returns a substring that should be valid JSON, or None.
        """
        if not raw:
            return None

        s = raw.strip()
        # Find first '{' or '['
        start = -1
        for i, ch in enumerate(s):
            if ch == "{" or ch == "[":
                start = i
                break
        if start < 0:
            return None

        stack = []
        in_str = False
        esc = False
        quote_ch = ""
        for j in range(start, len(s)):
            ch = s[j]

            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == quote_ch:
                    in_str = False
                    quote_ch = ""
                continue

            # Not in string
            if ch == '"' or ch == "'":
                in_str = True
                quote_ch = ch
                continue

            if ch == "{" or ch == "[":
                stack.append(ch)
                continue
            if ch == "}" or ch == "]":
                if not stack:
                    # Unbalanced close; ignore
                    continue
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        # Completed first JSON region
                        return s[start : j + 1].strip()
                else:
                    # Mismatched; keep scanning
                    continue

        return None

    @staticmethod
    def _json_loads_with_salvage(raw: str) -> tuple[Any | None, bool]:
        """
        Try json.loads(raw). If it fails, try salvage extraction and parse again.
        Returns (obj, salvaged_bool). obj=None if still failing.
        """
        if not raw:
            return None, False
        try:
            return json.loads(raw), False
        except Exception:
            pass

        extracted = LLMClient._extract_first_json(raw)
        if not extracted:
            return None, False
        try:
            return json.loads(extracted), True
        except Exception:
            return None, False

    def _max_attempts_for_kind(self, default_attempts: int, kind: LLMFailureKind | None) -> int:
        """
        Retry policy by failure kind.
        INVALID_JSON tends not to converge with repeated attempts on the same provider.
        """
        if kind == LLMFailureKind.INVALID_JSON:
            return 1
        return default_attempts

    def _model_supports_structured_outputs(self, model: str) -> bool:
        """Check if model supports OpenAI Structured Outputs (json_schema mode)."""
        # OpenAI models that support structured outputs
        # See: https://platform.openai.com/docs/guides/structured-outputs
        supported_prefixes = (
            "gpt-5",      # All GPT-5 variants
            "gpt-4o",     # GPT-4o and mini
            "gpt-4-turbo",
            "o1",         # o1 models
            "o3",         # o3 models
        )
        # DeepSeek and local models don't support structured outputs
        unsupported_prefixes = ("deepseek", "llama", "mistral", "qwen")
        
        model_lower = model.lower()
        if any(model_lower.startswith(p) for p in unsupported_prefixes):
            return False
        if any(model_lower.startswith(p) for p in supported_prefixes):
            return True
        # Default: try structured outputs for unknown OpenAI models
        return self.base_url is None  # Only for official OpenAI API

    async def _call_chat_completions(
        self,
        *,
        model: str,
        input: str,  # noqa: A002
        instructions: str | None = None,
        json_output: bool = False,
        response_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        stage: str | None = None,
    ) -> dict:
        """
        Execute LLM call using Chat Completions API for local endpoints.
        
        Does NOT send Responses-specific params (reasoning_effort, cache_key, etc.)
        that local endpoints (vLLM, TGI) do not support.
        """
        import hashlib
        import time

        effective_timeout = timeout or self.default_timeout
        json_output_requested = json_output or response_schema is not None

        # Build messages for Chat Completions API
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": input})

        # Build params - only standard Chat Completions fields
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": effective_timeout,
        }

        if temperature is not None:
            params["temperature"] = temperature

        if max_output_tokens:
            params["max_tokens"] = max_output_tokens  # Chat Completions uses max_tokens

        # JSON mode or Structured Outputs
        if json_output_requested:
            if response_schema and self._model_supports_structured_outputs(model):
                # Use OpenAI Structured Outputs with strict schema enforcement
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": trace_kind.replace(".", "_") if trace_kind else "response",
                        "schema": response_schema,
                        "strict": True,
                    }
                }
            else:
                # Fallback to simple JSON mode
                params["response_format"] = {"type": "json_object"}

        # Payload hash for tracing
        payload_str = (instructions or "") + "||" + input
        payload_hash = hashlib.md5(payload_str.encode()).hexdigest()

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
            "api_mode": "chat_completions",
            "base_url": self.base_url,
            "input_text": input[:10000],
            "instructions_text": (instructions or "")[:5000],
        })

        last_error: Exception | None = None
        effective_max_attempts = self.max_retries

        for attempt in range(self.max_retries):
            # Check effective retry limit based on error kind
            if attempt >= effective_max_attempts:
                break

            start_time = time.time()
            try:
                async with self._sem:
                    if attempt > 0:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        backoff_seconds = min(2 ** attempt, 30)  # Cap at 30s
                        await asyncio.sleep(backoff_seconds)
                        logger.debug("[LLMClient] Chat Completions retry %d/%d for %s (backoff: %ds)", attempt + 1, self.max_retries, model, backoff_seconds)

                    response = await self.client.chat.completions.create(**params)

                latency_ms = int((time.time() - start_time) * 1000)

                # Extract content from Chat Completions response
                if not response.choices:
                    raise ValueError("Empty response from LLM (no choices)")

                choice = response.choices[0]
                content = choice.message.content or ""

                if not content.strip():
                    finish_reason = choice.finish_reason
                    if finish_reason == "length":
                        raise ValueError("Response truncated (max_tokens reached)")
                    raise ValueError("Empty response from LLM")

                # Parse JSON if requested
                parsed = None
                if json_output_requested:
                    parsed, salvaged = self._json_loads_with_salvage(content)
                    
                    if parsed is None:
                        # Mark as INVALID_JSON for retry policy + upstream fallback
                        e = RuntimeError(f"LLM JSON parse failed. Content: {content[:200]}...")
                        Trace.event(
                            f"{trace_kind}.json_parse_failed",
                            {
                                "model": model,
                                "trace_key": trace_kind,
                                "attempt": attempt + 1,
                                "kind": "invalid_json",
                                "content_head": content[:500],
                                "payload_hash": payload_hash,
                            },
                        )
                        logger.warning("[LLMClient] Chat Completions JSON parse failed: raw not parseable (attempt=%s)", attempt + 1)
                        raise LLMCallError(
                            message=f"LLM JSON parse failed: {str(e)}",
                            kind=LLMFailureKind.INVALID_JSON,
                        ) from e

                    if salvaged:
                        Trace.event(
                            f"{trace_kind}.json_recovered_from_prefix",
                            {
                                "model": model,
                                "trace_key": trace_kind,
                                "attempt": attempt + 1,
                                "payload_hash": payload_hash,
                                "api_mode": "chat_completions",
                            },
                        )

                    # Validate against schema if provided
                    if response_schema and parsed is not None:
                        schema_errors = _validate_schema(parsed, response_schema)
                        if schema_errors:
                            logger.warning("[LLMClient] Schema validation failed: %s errors", len(schema_errors))
                            Trace.event(
                                f"{trace_kind}.schema_validation_error",
                                {
                                    "model": model,
                                    "errors": schema_errors[:10],
                                    "api_mode": "chat_completions",
                                    "payload_hash": payload_hash,
                                },
                            )
                            # Schema repair retry: ask model to fix the JSON
                            repair_prompt = (
                                "Your JSON output has schema errors. Fix them and return ONLY the corrected JSON.\n\n"
                                "ERRORS:\n" + "\n".join(f"- {e}" for e in schema_errors[:10]) + "\n\n"
                                f"YOUR JSON (fix it):\n{content}\n\n"
                                "Return ONLY the fixed JSON. No explanation. Start with {{"
                            )
                            try:
                                repair_messages = [{"role": "user", "content": repair_prompt}]
                                repair_params = {
                                    "model": model,
                                    "messages": repair_messages,
                                    "timeout": effective_timeout,
                                    "response_format": {"type": "json_object"},
                                }
                                if temperature is not None:
                                    repair_params["temperature"] = temperature
                                
                                Trace.event(f"{trace_kind}.schema_repair_attempt", {
                                    "model": model,
                                    "errors_count": len(schema_errors),
                                    "payload_hash": payload_hash,
                                })
                                
                                repair_response = await self.client.chat.completions.create(**repair_params)
                                repair_content = repair_response.choices[0].message.content or ""
                                
                                # Record repair call cost
                                try:
                                    meter = self._meter or get_current_llm_meter()
                                    if meter and repair_response.usage:
                                        repair_usage = {
                                            "input_tokens": repair_response.usage.prompt_tokens,
                                            "output_tokens": repair_response.usage.completion_tokens,
                                        }
                                        meter.record_completion(
                                            model=model,
                                            stage=f"{stage or trace_kind}_repair",
                                            usage=repair_usage,
                                            input_text=repair_prompt,
                                            output_text=repair_content,
                                            instructions=None,
                                        )
                                except Exception:
                                    pass  # Metering failure is non-critical
                                
                                # Try to parse repaired JSON
                                repaired_parsed, _ = self._json_loads_with_salvage(repair_content)
                                
                                if repaired_parsed is not None:
                                    # Validate repaired output
                                    repair_errors = _validate_schema(repaired_parsed, response_schema)
                                    if not repair_errors:
                                        Trace.event(f"{trace_kind}.schema_repair_success", {
                                            "model": model,
                                            "payload_hash": payload_hash,
                                        })
                                        parsed = repaired_parsed
                                        content = repair_content
                                    else:
                                        Trace.event(f"{trace_kind}.schema_repair_failed", {
                                            "model": model,
                                            "remaining_errors": repair_errors[:5],
                                            "payload_hash": payload_hash,
                                        })
                                        raise ValueError(f"LLM schema validation failed after repair: {repair_errors[0]}")
                                else:
                                    raise ValueError("LLM schema repair returned invalid JSON")
                            except Exception as repair_exc:
                                logger.warning("[LLMClient] Schema repair failed: %s", repair_exc)
                                Trace.event(f"{trace_kind}.schema_repair_exception", {
                                    "model": model,
                                    "error": str(repair_exc)[:200],
                                    "payload_hash": payload_hash,
                                })
                                raise ValueError(f"LLM schema validation failed: {schema_errors[0]}") from repair_exc

                # Extract usage info
                usage = None
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cached_tokens": 0,  # Chat Completions doesn't have caching
                        "latency_ms": latency_ms,
                    }
                else:
                    usage = {"latency_ms": latency_ms}

                result = {
                    "content": content,
                    "parsed": parsed,
                    "model": response.model if hasattr(response, "model") else model,
                    "cache_status": "NONE",  # No caching for local endpoints
                    "usage": usage,
                }

                # Metering
                meter = self._meter or get_current_llm_meter()
                if meter:
                    try:
                        event = meter.record_completion(
                            model=model,
                            stage=stage or trace_kind,
                            usage=usage,
                            input_text=input,
                            output_text=content,
                            instructions=instructions,
                        )
                        Trace.event("llm.metering.recorded", {
                            "stage": stage or trace_kind,
                            "cost_credits": str(event.cost_credits),
                            "api_mode": "chat_completions",
                        })
                    except Exception:
                        pass  # Metering failure is non-critical

                Trace.event(f"{trace_kind}.response", {
                    "model": model,
                    "content_chars": len(content),
                    "cache_status": "NONE",
                    "attempt": attempt + 1,
                    "payload_hash": payload_hash,
                    "api_mode": "chat_completions",
                    "response_text": content[:15000],
                })

                return result

            except Exception as e:
                last_error = e
                error_str = str(e)[:200]
                is_connection_error = "connection" in error_str.lower() or "timeout" in error_str.lower()
                logger.warning("[LLMClient] Chat Completions attempt %d failed: %s", attempt + 1, e)
                Trace.event(f"{trace_kind}.error", {
                    "model": model,
                    "attempt": attempt + 1,
                    "error": error_str,
                    "is_connection_error": is_connection_error,
                    "payload_hash": payload_hash,
                    "api_mode": "chat_completions",
                })

        # All retries exhausted - emit provider_down event
        Trace.event(f"{trace_kind}.provider_down", {
            "model": model,
            "total_attempts": self.max_retries,
            "last_error": str(last_error)[:200] if last_error else None,
            "api_mode": "chat_completions",
        })
        raise ValueError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

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
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        stage: str | None = None,
    ) -> dict:
        """
        Execute LLM call using Responses API (OpenAI) or Chat Completions API (local).
        
        When base_url is set (local endpoint), uses Chat Completions API and ignores
        Responses-specific parameters (reasoning_effort, cache_key, instructions as separate field).
        
        Args:
            model: Model to use (e.g., "gpt-5-nano", "qwen3-base-14b")
            input: The main input/prompt content
            instructions: System instructions (cached if cache_key provided for OpenAI)
            json_output: If True, request JSON output and parse response
            response_schema: Optional JSON schema for structured output
            reasoning_effort: Reasoning effort level (low/medium/high) - OpenAI only
            cache_key: Optional key for prompt caching - OpenAI only
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
        # Route to Chat Completions API for local endpoints
        if self._use_chat_completions:
            return await self._call_chat_completions(
                model=model,
                input=input,
                instructions=instructions,
                json_output=json_output,
                response_schema=response_schema,
                timeout=timeout,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                trace_kind=trace_kind,
                stage=stage,
            )

        effective_timeout = timeout or self.default_timeout

        json_output_requested = json_output or response_schema is not None

        # Build API params for Responses API (OpenAI)
        params: dict = {
            "model": model,
            "input": input,
            "timeout": effective_timeout,
        }

        if temperature is not None:
            # Skip temperature for models that don't support it in Responses API
            # Based on empirical testing: gpt-5-nano, gpt-5, gpt-5.2, and O-series models reject temperature
            skip_temp_models = model.startswith("o") or model.startswith("gpt-5")
            if skip_temp_models:
                logger.debug("[LLMClient] Temperature ignored for model %s (not supported)", model)
            else:
                params["temperature"] = temperature

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
                # Note: strict=False because our schemas have optional fields
                # (strict=True requires ALL properties to be in required)
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": False,
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

        # Calculate payload hash for debug correlation
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
        effective_max_attempts = self.max_retries

        for attempt in range(self.max_retries):
            # Check effective retry limit based on error kind
            if attempt >= effective_max_attempts:
                break

            start_time = time.time()
            try:
                async with self._sem:
                    if attempt > 0:
                        # Exponential backoff: 1s, 2s, 4s, ...
                        backoff_seconds = min(2 ** attempt, 30)  # Cap at 30s
                        await asyncio.sleep(backoff_seconds)
                        logger.debug("[LLMClient] Retry %d/%d for %s (backoff: %ds)", attempt + 1, self.max_retries, model, backoff_seconds)

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
                    parsed, salvaged = self._json_loads_with_salvage(content)
                    
                    if parsed is None:
                         # Mark as INVALID_JSON for retry policy + upstream fallback
                        e = RuntimeError(f"LLM JSON parse failed. Content: {content[:200]}...")
                        Trace.event(
                            f"{trace_kind}.json_parse_failed",
                            {
                                "model": model,
                                "trace_key": trace_kind,
                                "attempt": attempt + 1,
                                "kind": "invalid_json",
                                "content_head": content[:500],
                                "payload_hash": payload_hash,
                            },
                        )
                        logger.warning("[LLMClient] JSON parse failed: raw not parseable (attempt=%s)", attempt + 1)
                        raise LLMCallError(
                            message=f"LLM JSON parse failed: {str(e)}",
                            kind=LLMFailureKind.INVALID_JSON,
                        ) from e

                    if salvaged:
                        Trace.event(
                            f"{trace_kind}.json_recovered_from_prefix",
                            {
                                "model": model,
                                "trace_key": trace_kind,
                                "attempt": attempt + 1,
                                "payload_hash": payload_hash,
                            },
                        )

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

                meter = self._meter or get_current_llm_meter()
                if meter:
                    try:
                        event = meter.record_completion(
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
                        Trace.event("llm.metering.failed", {"error": str(exc)[:200]})
                # Skip metering trace - no actionable info (removed for trace size reduction)

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

            except LLMCallError as e:
                last_error = e
                # Adjust effective attempts after first typed failure
                if attempt == 0:
                    effective_max_attempts = self._max_attempts_for_kind(self.max_retries, e.kind)
                logger.warning("[LLMClient] Attempt %d failed: %s", attempt + 1, e)
                if attempt + 1 >= effective_max_attempts:
                     Trace.event(f"{trace_kind}.error", {
                        "model": model,
                        "attempt": attempt + 1,
                        "error": str(e),
                        "kind": e.kind.value,
                        "payload_hash": payload_hash
                    })
               
            except Exception as e:
                last_error = e
                # Unknown errors: keep default retry budget
                logger.warning("[LLMClient] Attempt %d failed: %s", attempt + 1, e)
                if attempt + 1 >= effective_max_attempts:
                     error_str = str(e)[:200]
                     is_connection_error = "connection" in error_str.lower() or "timeout" in error_str.lower()
                     Trace.event(f"{trace_kind}.error", {
                        "model": model,
                        "attempt": attempt + 1,
                        "error": error_str,
                        "is_connection_error": is_connection_error,
                        "payload_hash": payload_hash
                    })


        # All retries exhausted - emit provider_down event
        Trace.event(f"{trace_kind}.provider_down", {
            "model": model,
            "total_attempts": self.max_retries,
            "last_error": str(last_error)[:200] if last_error else None,
            "api_mode": "responses",
        })
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
        temperature: float | None = None,
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
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
        )
        return result["parsed"]

    async def call_structured(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        schema: dict[str, Any],
        schema_name: str = "structured_output",
        model: str = "gpt-5-nano",
        reasoning_effort: ReasoningEffort = "low",
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        trace_kind: str = "llm_call",
        temperature: float | None = None,
    ) -> dict:
        """
        Structured output call with JSON schema.
        
        This is a convenience wrapper around call_json that uses a specific
        parameter naming convention used by skills.
        
        Args:
            user_prompt: The user message content
            system_prompt: Optional system instructions
            schema: JSON schema for structured output
            schema_name: Name for the schema
            model: Model to use (default: gpt-5-nano)
            temperature: Optional temperature for sampling (0 = deterministic)
            
        Returns:
            Parsed JSON response matching the schema
        """
        # Ensure schema has title for the API
        schema_with_title = dict(schema)
        if "title" not in schema_with_title:
            schema_with_title["title"] = schema_name
            
        return await self.call_json(
            model=model,
            input=user_prompt,
            instructions=system_prompt,
            response_schema=schema_with_title,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            trace_kind=trace_kind,
            temperature=temperature,
        )

    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.close()