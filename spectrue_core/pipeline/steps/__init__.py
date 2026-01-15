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
from .claim_cluster import ClaimClusterStep
from .claim_clusters import ClaimClustersStep
from .target_selection import TargetSelectionStep
from .evaluate_semantic_gating import EvaluateSemanticGatingStep
from .retrieval import (
    AssembleRetrievalItemsStep,
    BuildQueriesStep,
    BuildClusterQueriesStep,
    ClusterEvidenceEnrichStep,
    FetchChunksStep,
    ClusterWebSearchStep,
    RerankStep,
    WebSearchStep,
    ClusterAttributionStep,
)
from .evidence_collect import EvidenceCollectStep
from .evidence_gating import EvidenceGatingStep
from .stance_annotate import StanceAnnotateStep
from .cluster_evidence import ClusterEvidenceStep
from .audit_claims import AuditClaimsStep
from .audit_evidence import AuditEvidenceStep
from .aggregate_rgba_audit import AggregateRGBAAuditStep
from .judge_standard import JudgeStandardStep
from .oracle_flow import OracleFlowStep
from .result_assembly import AssembleStandardResultStep, ResultAssemblyStep
from .extraction_result_assembly import ExtractionResultAssemblyStep
from .cost_summary import CostSummaryStep
from .invariants import (
    AssertNonEmptyClaimsStep,
    AssertContractPresenceStep,
    AssertSingleClaimStep,
    AssertSingleLanguageStep,
    AssertStandardResultKeysStep,
    AssertCostNonZeroStep,
    AssertRetrievalTraceStep,
    AssertDeepJudgingStep,
)

__all__ = [
    "MeteringSetupStep",
    "PrepareInputStep",
    "ExtractClaimsStep",
    "VerifyInlineSourcesStep",
    "ClaimGraphStep",
    "ClaimClusterStep",
    "ClaimClustersStep",
    "TargetSelectionStep",
    "EvaluateSemanticGatingStep",
    "BuildQueriesStep",
    "BuildClusterQueriesStep",
    "WebSearchStep",
    "ClusterWebSearchStep",
    "ClusterEvidenceEnrichStep",
    "RerankStep",
    "FetchChunksStep",
    "AssembleRetrievalItemsStep",
    "ClusterAttributionStep",
    "EvidenceCollectStep",
    "EvidenceGatingStep",
    "StanceAnnotateStep",
    "ClusterEvidenceStep",
    "AuditClaimsStep",
    "AuditEvidenceStep",
    "AggregateRGBAAuditStep",
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
    "AssertStandardResultKeysStep",
    "AssertCostNonZeroStep",
    "AssertRetrievalTraceStep",
    "AssertDeepJudgingStep",
]
