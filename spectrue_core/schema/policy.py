# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M71: Verdict Policy & Enums.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel


class ErrorState(str, Enum):
    """Pipeline/sensor failure states."""
    
    OK = "ok"
    NO_EVIDENCE_RETRIEVED = "no_evidence_retrieved"
    EVIDENCE_UNREADABLE = "evidence_unreadable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_RATE_LIMIT = "upstream_rate_limit"
    PIPELINE_ERROR = "pipeline_error"


class DecisionPath(str, Enum):
    """How the verdict was reached."""
    
    ORACLE = "oracle"
    WEB = "web"
    CACHE = "cache"


class VerdictPolicy(SchemaModel):
    """Configuration-driven thresholds for verdict derivation."""
    
    # Confidence thresholds (stricter than before: 0.8)
    min_confidence_for_verified: float = Field(default=0.8, ge=0.0, le=1.0)
    min_confidence_for_refuted: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Veracity thresholds
    verified_veracity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    refuted_veracity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Error Handling
    unknown_if_error_states: set[ErrorState] = Field(
        default_factory=lambda: {
            ErrorState.NO_EVIDENCE_RETRIEVED,
            ErrorState.EVIDENCE_UNREADABLE,
            ErrorState.UPSTREAM_TIMEOUT,
            ErrorState.PIPELINE_ERROR,
        }
    )
    
    # Evidence Rules
    min_quote_len_chars: int = Field(default=20, ge=0)
    
    # Safety Ceiling: If no quotes found, cap confidence at 0.5 (forces Ambiguous)
    max_confidence_without_quotes: float | None = Field(default=0.5, ge=0.0, le=1.0)
    

# Default policy for general fact-checking
DEFAULT_POLICY = VerdictPolicy()
