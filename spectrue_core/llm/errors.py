from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class LLMFailureKind(str, Enum):
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION = "schema_validation"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


@dataclass
class LLMCallError(Exception):
    message: str
    kind: LLMFailureKind = LLMFailureKind.UNKNOWN

    def __str__(self) -> str:
        return f"{self.message} (kind={self.kind.value})"
