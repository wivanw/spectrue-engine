# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""
Decomposed Pipeline Steps.

One class per file.
"""

from .metering_setup import MeteringSetupStep
from .prepare_input import PrepareInputStep
from .extract_claims import ExtractClaimsStep
from .verify_inline_sources import VerifyInlineSourcesStep
from .claim_graph import ClaimGraphStep
from .target_selection import TargetSelectionStep
from .evaluate_semantic_gating import EvaluateSemanticGatingStep
from .search_flow import SearchFlowStep
from .evidence_collect import EvidenceCollectStep
from .evidence_flow import EvidenceFlowStep
from .judge_standard import JudgeStandardStep
from .oracle_flow import OracleFlowStep
from .result_assembly import AssembleStandardResultStep, ResultAssemblyStep
from .extraction_result_assembly import ExtractionResultAssemblyStep
from .cost_summary import CostSummaryStep
# M119: Add invariant steps
from .invariants import (
    AssertNonEmptyClaimsStep,
    AssertContractPresenceStep,
    AssertSingleClaimStep,
    AssertSingleLanguageStep,
)

__all__ = [
    "MeteringSetupStep",
    "PrepareInputStep",
    "ExtractClaimsStep",
    "VerifyInlineSourcesStep",
    "ClaimGraphStep",
    "TargetSelectionStep",
    "EvaluateSemanticGatingStep",
    "SearchFlowStep",
    "EvidenceCollectStep",
    "EvidenceFlowStep",
    "JudgeStandardStep",
    "OracleFlowStep",
    "AssembleStandardResultStep",
    "ResultAssemblyStep",
    "ExtractionResultAssemblyStep",
    "CostSummaryStep",
    # Invariants
    "AssertNonEmptyClaimsStep",
    "AssertContractPresenceStep",
    "AssertSingleClaimStep",
    "AssertSingleLanguageStep",
]
