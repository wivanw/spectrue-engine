from spectrue_core.llm.model_registry import ModelID
# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
LLM Provider Fallback Wrapper.

Implements resilient LLM calling with automatic fallback to a secondary provider
when the primary provider fails due to connection errors, timeouts, or invalid output.
"""

import logging
from typing import Callable, Awaitable

from spectrue_core.utils.trace import Trace
from spectrue_core.llm.failures import (
    classify_llm_failure,
    is_fallback_eligible,
    failure_kind_to_trace_data,
)

logger = logging.getLogger(__name__)


async def call_with_fallback(
    *,
    primary_call: Callable[[], Awaitable[dict]],
    fallback_call: Callable[[], Awaitable[dict]],
    task_name: str,
    primary_provider_name: str = "primary",
    fallback_provider_name: str = "fallback",
) -> dict:
    """
    Call primary provider, falling back to secondary if it fails.
    
    Args:
        primary_call: Async callable for the primary provider
        fallback_call: Async callable for the fallback provider
        task_name: Name of the task for tracing (e.g., "claims.extraction")
        primary_provider_name: Name of primary provider (e.g., "deepseek")
        fallback_provider_name: Name of fallback provider (e.g., ModelID.PRO)
        
    Returns:
        Result dict from the successful provider
        
    Raises:
        Exception: The last exception encountered if both calls fail
    """
    try:
        # Attempt primary call
        result = await primary_call()
        
        Trace.event("llm.call.ok", {
            "task": task_name,
            "provider": primary_provider_name,
            "is_fallback": False,
        })
        
        return result
        
    except Exception as e:
        # Check if error is eligible for fallback
        if not is_fallback_eligible(e):
            # Non-retriable error (e.g. business logic error if we defined any)
            # But currently we treat all LLM exceptions as eligible
            raise e
            
        failure_kind = classify_llm_failure(e)
        
        logger.warning(
            "[Fallback] Primary provider %s failed for %s: %s (kind=%s). Switching to %s.",
            primary_provider_name, task_name, e, failure_kind, fallback_provider_name
        )
        
        # Trace failure
        trace_data = {
            "task": task_name,
            "provider": primary_provider_name,
            **failure_kind_to_trace_data(failure_kind, e)
        }
        Trace.event("llm.call.failed", trace_data)
        
        # Emit fallback usage event
        Trace.event("llm.fallback.used", {
            "task": task_name,
            "primary_provider": primary_provider_name,
            "fallback_provider": fallback_provider_name,
            "failure_kind": failure_kind.value if failure_kind else "unknown",
            "primary_error": str(e)[:200],
        })
        
        try:
            # Attempt fallback call
            result = await fallback_call()
            
            Trace.event("llm.call.ok", {
                "task": task_name,
                "provider": fallback_provider_name,
                "is_fallback": True,
                "fallback_from": primary_provider_name,
            })
            
            return result
            
        except Exception as fallback_e:
            logger.error(
                "[Fallback] Fallback provider %s also failed for %s: %s",
                fallback_provider_name, task_name, fallback_e
            )
            
            # Trace complete failure
            fallback_kind = classify_llm_failure(fallback_e)
            Trace.event("llm.fallback.failed", {
                "task": task_name,
                "primary_provider": primary_provider_name,
                "fallback_provider": fallback_provider_name,
                "primary_error": str(e)[:200],
                "fallback_error": str(fallback_e)[:200],
                "fallback_failure_kind": fallback_kind.value if fallback_kind else "unknown",
            })
            
            # Re-raise the fallback exception as it's the final result
            raise fallback_e
