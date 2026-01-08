# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
LLM Failure Classification.

Defines failure types that trigger provider fallback:
- CONNECTION_ERROR: Network/connection issues
- TIMEOUT: Request timeout
- PROVIDER_ERROR: Provider returned error (5xx, rate limit)
- INVALID_JSON: Response not valid JSON
- SCHEMA_VALIDATION_FAILED: JSON doesn't match required schema
"""

from enum import Enum
from typing import Any


class LLMFailureKind(Enum):
    """Classification of LLM call failures that trigger fallback."""
    
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


# Keywords that indicate connection errors
_CONNECTION_KEYWORDS = (
    "connection",
    "connect",
    "network",
    "socket",
    "refused",
    "unreachable",
    "dns",
    "ssl",
    "certificate",
)

# Keywords that indicate timeout errors
_TIMEOUT_KEYWORDS = (
    "timeout",
    "timed out",
    "deadline exceeded",
)

# Keywords that indicate provider errors
_PROVIDER_ERROR_KEYWORDS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota",
    "capacity",
    "overloaded",
    "unavailable",
    "service error",
    "internal server",
    "500",
    "502",
    "503",
    "504",
)

# Keywords that indicate JSON parsing errors
_JSON_ERROR_KEYWORDS = (
    "json",
    "parse",
    "decode",
    "invalid character",
    "unexpected token",
    "expecting",
)

# Keywords that indicate schema validation errors
_SCHEMA_ERROR_KEYWORDS = (
    "schema",
    "validation failed",
    "missing required",
    "unexpected field",
    "expected object",
    "expected array",
    "expected string",
    "does not match",
)


def classify_llm_failure(exc: Exception) -> LLMFailureKind | None:
    """
    Classify an LLM call exception into a failure kind.
    
    Args:
        exc: The exception raised during LLM call
        
    Returns:
        LLMFailureKind if classifiable, None if not a recognized failure
    """
    error_msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    
    # Check for provider errors (Highest priority to catch "Rate limit exceeded")
    if any(kw in error_msg for kw in _PROVIDER_ERROR_KEYWORDS):
        return LLMFailureKind.PROVIDER_ERROR
        
    # Check for JSON parsing errors
    if any(kw in error_msg for kw in _JSON_ERROR_KEYWORDS):
        return LLMFailureKind.INVALID_JSON
    if "json" in exc_type:
        return LLMFailureKind.INVALID_JSON
    
    # Check for schema validation errors
    if any(kw in error_msg for kw in _SCHEMA_ERROR_KEYWORDS):
        return LLMFailureKind.SCHEMA_VALIDATION_FAILED
    
    # Check for timeout
    if any(kw in error_msg for kw in _TIMEOUT_KEYWORDS):
        return LLMFailureKind.TIMEOUT
    if "timeout" in exc_type:
        return LLMFailureKind.TIMEOUT
    
    # Check for connection errors
    if any(kw in error_msg for kw in _CONNECTION_KEYWORDS):
        return LLMFailureKind.CONNECTION_ERROR
    if "connection" in exc_type or "network" in exc_type:
        return LLMFailureKind.CONNECTION_ERROR
    
    # Unknown failure type - still a failure but not classified
    return None


def is_fallback_eligible(exc: Exception) -> bool:
    """
    Check if an exception should trigger fallback.
    
    All classified failures are fallback-eligible.
    Unclassified exceptions are also eligible (conservative approach).
    """
    # Any exception during LLM call should trigger fallback
    return True


def failure_kind_to_trace_data(kind: LLMFailureKind | None, exc: Exception) -> dict[str, Any]:
    """
    Convert failure info to trace event data.
    """
    return {
        "failure_kind": kind.value if kind else "unknown",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:200],
    }
