from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class LLMFailureKind(str, Enum):
    """Classification of LLM call failures."""
    
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    RATE_LIMITED = "rate_limited"
    INPUT_TOO_LONG = "input_too_long"
    UNSUPPORTED_ROUTE = "unsupported_route"
    UNKNOWN = "unknown"


@dataclass
class LLMCallError(Exception):
    message: str
    kind: LLMFailureKind = LLMFailureKind.UNKNOWN

    def __str__(self) -> str:
        return f"{self.message} (kind={self.kind.value})"
