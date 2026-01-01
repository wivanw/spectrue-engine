# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Pipeline Steps Module

Exports all available step implementations.

M118: Legacy steps removed. All steps are now DAG-native.
"""

from spectrue_core.pipeline.steps.invariants import (
    AssertSingleClaimStep,
    AssertSingleLanguageStep,
    AssertNonEmptyClaimsStep,
    AssertMaxClaimsStep,
    AssertMeteringEnabledStep,
    get_invariant_steps_for_mode,
)
from spectrue_core.pipeline.steps.decomposed import (
    MeteringSetupStep,
    PrepareInputStep,
    ExtractClaimsStep,
    ClaimGraphStep,
    TargetSelectionStep,
    SearchFlowStep,
    EvidenceFlowStep,
    OracleFlowStep,
    ResultAssemblyStep,
)
from spectrue_core.pipeline.steps.deep_claim import (
    BuildClaimFramesStep,
    SummarizeEvidenceStep,
    JudgeClaimsStep,
    AssembleDeepResultStep,
)

__all__ = [
    # Invariant Steps
    "AssertSingleClaimStep",
    "AssertSingleLanguageStep",
    "AssertNonEmptyClaimsStep",
    "AssertMaxClaimsStep",
    "AssertMeteringEnabledStep",
    "get_invariant_steps_for_mode",
    # Decomposed Steps (M115)
    "MeteringSetupStep",
    "PrepareInputStep",
    "ExtractClaimsStep",
    "ClaimGraphStep",
    "TargetSelectionStep",
    "SearchFlowStep",
    "EvidenceFlowStep",
    "OracleFlowStep",
    "ResultAssemblyStep",
    # Deep Claim Steps (M117)
    "BuildClaimFramesStep",
    "SummarizeEvidenceStep",
    "JudgeClaimsStep",
    "AssembleDeepResultStep",
]
