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
from .evidence_flow import EvidenceFlowStep
from .oracle_flow import OracleFlowStep
from .result_assembly import ResultAssemblyStep
from .extraction_result_assembly import ExtractionResultAssemblyStep
# Deep steps (assuming they are already in their own files or need similar treatment?)
# Checking deep_claim.py existence or content.
# Assuming deep_claim.py is already fine or I should check it.
# For now, re-exporting the ones I extracted.

__all__ = [
    "MeteringSetupStep",
    "PrepareInputStep",
    "ExtractClaimsStep",
    "VerifyInlineSourcesStep",
    "ClaimGraphStep",
    "TargetSelectionStep",
    "EvaluateSemanticGatingStep",
    "SearchFlowStep",
    "EvidenceFlowStep",
    "OracleFlowStep",
    "ResultAssemblyStep",
    "ExtractionResultAssemblyStep",
]
