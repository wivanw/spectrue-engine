# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M70: Schema-First Pipeline - Core Schema Module

This module provides Pydantic models for structured claim representation.
The schema is the STABLE CONTRACT between pipeline stages:
- Claim Extraction (Spec Producer) outputs ClaimUnit
- Search, Evidence, Scoring (Spec Consumers) consume structured fields

Philosophy:
- Schema = Structure (Python validates types/fields)
- LLM = All Decisions (dimension classification, scoring, stance)
- NO HEURISTICS in code
"""

from spectrue_core.schema.claims import (
    # Enums
    Dimension,
    VerificationScope,
    ClaimDomain,
    ClaimType,
    # Core models
    Assertion,
    SourceSpan,
    EvidenceRequirementSpec,
    LocationQualifier,
    EventRules,
    BroadcastInfo,
    EventQualifiers,
    ClaimUnit,
)

from spectrue_core.schema.evidence import (
    # Enums
    EvidenceStance,
    ContentStatus,
    # Models
    EvidenceItem,
)

from spectrue_core.schema.verdict import (
    # Enums
    VerdictStatus,
    # Models
    AssertionVerdict,
    ClaimVerdict,
    StructuredDebug,
    StructuredVerdict,
)

__all__ = [
    # Enums
    "Dimension",
    "VerificationScope", 
    "ClaimDomain",
    "ClaimType",
    "EvidenceStance",
    "ContentStatus",
    # Claim models
    "Assertion",
    "SourceSpan",
    "EvidenceRequirementSpec",
    "LocationQualifier",
    "EventRules",
    "BroadcastInfo",
    "EventQualifiers",
    "ClaimUnit",
    # Evidence models
    "EvidenceItem",
    # Verdict models
    "VerdictStatus",
    "AssertionVerdict",
    "ClaimVerdict",
    "StructuredDebug",
    "StructuredVerdict",
]
