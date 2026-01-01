# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Spectrue Core Schema Module

Milestones:
- Schema-First Pipeline (ClaimUnit, Assertion, StructuredVerdict)
- Verdict Data Contract (Verdict, EvidenceSignals, Policy)
"""

# Claims
from spectrue_core.schema.claims import (
    Dimension,
    VerificationScope,
    ClaimDomain,
    ClaimType,
    ClaimStructureType,
    ClaimStructure,
    Assertion,
    SourceSpan,
    EvidenceRequirementSpec,
    LocationQualifier,
    EventRules,
    BroadcastInfo,
    EventQualifiers,
    ClaimUnit,
)

# Evidence
from spectrue_core.schema.evidence import (
    EvidenceStance,
    ContentStatus,
    EvidenceItem,
    TimelinessStatus,
)

# Verdict (keep as primary for backward compat)
from spectrue_core.schema.verdict import (
    VerdictStatus,  # VerdictStatus (has PARTIALLY_VERIFIED)
    VerdictState,
    AssertionVerdict,
    ClaimVerdict,
    StructuredDebug,
    StructuredVerdict,
)

# Policy
from spectrue_core.schema.policy import (
    ErrorState,
    DecisionPath,
    VerdictPolicy,
    DEFAULT_POLICY,
)

# Signals
from spectrue_core.schema.signals import (
    RetrievalSignals,
    CoverageSignals,
    TimelinessSignals,
    EvidenceSignals,
    TimeGranularity,
    TimeWindow,
    LocaleDecision,
)

# Verdict Contract (new contract, explicit import)
from spectrue_core.schema.verdict_contract import (
    VerdictStatus as ContractVerdictStatus,
    VerdictHighlight,
    Verdict,
)

# Claim Metadata for Orchestration
from spectrue_core.schema.claim_metadata import (
    ClaimRole,
    VerificationTarget,
    EvidenceChannel,
    UsePolicy,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    ClaimMetadata,
    default_claim_metadata,
)

# Per-Claim Judging (Deep Analysis)
from spectrue_core.schema.claim_frame import (
    ContextExcerpt,
    ContextMeta,
    EvidenceItemFrame,
    EvidenceStats,
    RetrievalHop,
    RetrievalTrace,
    ClaimFrame,
    EvidenceReference,
    EvidenceSummary,
    RGBAScore,
    JudgeOutput,
    ClaimResult,
    DeepAnalysisResult,
)


__all__ = [
    # Claims
    "Dimension", "VerificationScope", "ClaimDomain", "ClaimType",
    "ClaimStructureType", "ClaimStructure",
    "Assertion", "SourceSpan", "EvidenceRequirementSpec",
    "LocationQualifier", "EventRules", "BroadcastInfo", "EventQualifiers",
    "ClaimUnit",
    # Evidence
    "EvidenceStance", "ContentStatus", "EvidenceItem",
    "TimelinessStatus",
    # Verdict (backward compat)
    "VerdictStatus", "VerdictState", "AssertionVerdict", "ClaimVerdict",
    "StructuredDebug", "StructuredVerdict",
    # Policy
    "ErrorState", "DecisionPath", "VerdictPolicy", "DEFAULT_POLICY",
    # Signals
    "RetrievalSignals", "CoverageSignals", "TimelinessSignals", "EvidenceSignals",
    "TimeGranularity", "TimeWindow", "LocaleDecision",
    # Verdict Contract
    "ContractVerdictStatus", "VerdictHighlight", "Verdict",
    # Claim Metadata
    "ClaimRole", "VerificationTarget", "EvidenceChannel", "UsePolicy",
    "MetadataConfidence", "SearchLocalePlan", "RetrievalPolicy",
    "ClaimMetadata", "default_claim_metadata",
    # Per-Claim Judging (Deep Analysis)
    "ContextExcerpt", "ContextMeta", "EvidenceItemFrame", "EvidenceStats",
    "RetrievalHop", "RetrievalTrace", "ClaimFrame",
    "EvidenceReference", "EvidenceSummary", "RGBAScore",
    "JudgeOutput", "ClaimResult", "DeepAnalysisResult",
]
