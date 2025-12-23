# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Spectrue Core Schema Module

Milestones:
- M70: Schema-First Pipeline (ClaimUnit, Assertion, StructuredVerdict)
- M71: Verdict Data Contract (Verdict, EvidenceSignals, Policy)
"""

# M70: Claims
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

# M70: Evidence
from spectrue_core.schema.evidence import (
    EvidenceStance,
    ContentStatus,
    EvidenceItem,
    TimelinessStatus,
)

# M70: Verdict (keep as primary for backward compat)
from spectrue_core.schema.verdict import (
    VerdictStatus,  # M70 VerdictStatus (has PARTIALLY_VERIFIED)
    VerdictState,
    AssertionVerdict,
    ClaimVerdict,
    StructuredDebug,
    StructuredVerdict,
)

# M71: Policy
from spectrue_core.schema.policy import (
    ErrorState,
    DecisionPath,
    VerdictPolicy,
    DEFAULT_POLICY,
)

# M71: Signals
from spectrue_core.schema.signals import (
    RetrievalSignals,
    CoverageSignals,
    TimelinessSignals,
    EvidenceSignals,
    TimeGranularity,
    TimeWindow,
    LocaleDecision,
)

# M71: Verdict Contract (new contract, explicit import)
from spectrue_core.schema.verdict_contract import (
    VerdictStatus as ContractVerdictStatus,
    VerdictHighlight,
    Verdict,
)

# M80: Claim Metadata for Orchestration
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


__all__ = [
    # M70: Claims
    "Dimension", "VerificationScope", "ClaimDomain", "ClaimType",
    "ClaimStructureType", "ClaimStructure",
    "Assertion", "SourceSpan", "EvidenceRequirementSpec",
    "LocationQualifier", "EventRules", "BroadcastInfo", "EventQualifiers",
    "ClaimUnit",
    # M70: Evidence
    "EvidenceStance", "ContentStatus", "EvidenceItem",
    "TimelinessStatus",
    # M70: Verdict (backward compat)
    "VerdictStatus", "VerdictState", "AssertionVerdict", "ClaimVerdict",
    "StructuredDebug", "StructuredVerdict",
    # M71: Policy
    "ErrorState", "DecisionPath", "VerdictPolicy", "DEFAULT_POLICY",
    # M71: Signals
    "RetrievalSignals", "CoverageSignals", "TimelinessSignals", "EvidenceSignals",
    "TimeGranularity", "TimeWindow", "LocaleDecision",
    # M71: Verdict Contract
    "ContractVerdictStatus", "VerdictHighlight", "Verdict",
    # M80: Claim Metadata
    "ClaimRole", "VerificationTarget", "EvidenceChannel", "UsePolicy",
    "MetadataConfidence", "SearchLocalePlan", "RetrievalPolicy",
    "ClaimMetadata", "default_claim_metadata",
]
