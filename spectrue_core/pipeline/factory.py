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
Pipeline Factory

Builds DAGPipeline instances with the correct step composition for each mode.
This is the single place where mode -> step sequence mapping is defined.

M118: Removed legacy steps. All pipelines now use DAG architecture with
decomposed steps. No more `if(deep)` conditionals in pipeline code.

Usage:
    from spectrue_core.pipeline.factory import PipelineFactory

    factory = PipelineFactory(search_mgr=search_mgr, agent=agent)
    pipeline = factory.build("normal", config=config, pipeline=validation_pipeline)
    result = await pipeline.run(ctx)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spectrue_core.pipeline.dag import DAGPipeline

from spectrue_core.pipeline.mode import get_mode
from spectrue_core.pipeline.steps.invariants import (
    AssertNonEmptyClaimsStep,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineFactory:
    """
    Factory for building DAGPipeline instances.

    The factory is the ONLY place where mode -> step sequence is defined.
    This eliminates scattered if-statements throughout the codebase.

    M118: All methods now build DAGPipeline. Legacy Pipeline class is deprecated.

    Attributes:
        search_mgr: SearchManager for retrieval steps
        agent: FactCheckerAgent for LLM steps
    """

    search_mgr: Any  # SearchManager
    agent: Any  # FactCheckerAgent
    claim_graph: Any | None = None  # ClaimGraphBuilder (optional)

    def build(
        self,
        mode_name: str = "normal", # Default if not specified, though caller usually specifies
        *,
        config: Any,
        extraction_only: bool = False,
    ) -> "DAGPipeline":
        """
        Build a DAGPipeline with decomposed steps.

        This is the primary entry point for pipeline construction.
        All mode logic is encapsulated here.

        Args:
            mode_name: "normal", "general", or "deep"
            config: SpectrueConfig

        Returns:
            DAGPipeline configured with decomposed steps for the mode
        """
        from spectrue_core.pipeline.dag import DAGPipeline

        mode = get_mode(mode_name)

        if extraction_only:
            mode_name = "normal" # Default mode for context
            mode = get_mode(mode_name) # Ensure mode object exists
            nodes = self._build_extraction_dag_nodes(config)
        elif mode.name == "normal":
            nodes = self._build_normal_dag_nodes(config)
        else:
            nodes = self._build_deep_dag_nodes(config)

        logger.debug(
            "[PipelineFactory] Built DAG pipeline with %d nodes: %s",
            len(nodes),
            [n.name for n in nodes],
        )

        return DAGPipeline(mode=mode, nodes=nodes)

    def build_from_profile(self, profile_name: str, *, config: Any) -> "DAGPipeline":
        """
        Build pipeline from a profile name.

        Maps profile names to modes:
        - "normal" -> normal mode
        - "general" -> normal mode
        - "deep" -> deep mode

        This allows integration with the existing pipeline_builder module.
        """
        mode_mapping = {
            "normal": "normal",
            "general": "normal",
            "deep": "deep",
        }

        mode_name = mode_mapping.get(profile_name.lower(), "normal")
        return self.build(mode_name, config=config)

    def _build_normal_dag_nodes(self, config: Any) -> list:
        """Build DAG nodes for normal mode."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps import (
            ClaimGraphStep,
            EvidenceCollectStep,
            EvidenceGatingStep,
            StanceAnnotateStep,
            ClusterEvidenceStep,
            ExtractClaimsStep,
            JudgeStandardStep,
            MeteringSetupStep,
            OracleFlowStep,
            PrepareInputStep,
            AssembleStandardResultStep,
            EvaluateSemanticGatingStep,
            BuildQueriesStep,
            WebSearchStep,
            RerankStep,
            FetchChunksStep,
            AssembleRetrievalItemsStep,
            TargetSelectionStep,
            VerifyInlineSourcesStep,
            CostSummaryStep,
            AssertStandardResultKeysStep,
            AssertRetrievalTraceStep,
        )

        # All steps now always included (no feature flags)
        # Optional=True allows graceful failure without blocking pipeline

        return [
            # ==================== INFRASTRUCTURE ====================
            StepNode(step=MeteringSetupStep(config=config, agent=self.agent, search_mgr=self.search_mgr)),

            # ==================== INPUT PREPARATION ====================
            StepNode(
                step=PrepareInputStep(agent=self.agent, search_mgr=self.search_mgr, config=config),
                depends_on=["metering_setup"],
            ),

            # ==================== CLAIM EXTRACTION ====================
            StepNode(
                step=ExtractClaimsStep(agent=self.agent),
                depends_on=["prepare_input"],
            ),

            # Post-extraction invariant
            StepNode(
                step=AssertNonEmptyClaimsStep(),
                depends_on=["extract_claims"],
            ),

            # ==================== PARALLEL: INLINE + GATING + GRAPH ====================
            StepNode(
                step=VerifyInlineSourcesStep(agent=self.agent, search_mgr=self.search_mgr, config=config),
                depends_on=["extract_claims"],
            ),
            StepNode(
                step=EvaluateSemanticGatingStep(agent=self.agent),
                depends_on=["extract_claims"],
                optional=True,
            ),
            StepNode(
                step=ClaimGraphStep(
                    claim_graph=self.claim_graph,
                    runtime_config=config.runtime if hasattr(config, 'runtime') else None,
                ),
                depends_on=["extract_claims"],
            ),

            # Oracle fast path (optional)
            StepNode(
                step=OracleFlowStep(search_mgr=self.search_mgr),
                depends_on=["extract_claims"],
                optional=True,
            ),

            # Target selection (anchor-based for normal mode)
            StepNode(
                step=TargetSelectionStep(process_all_claims=False),
                depends_on=["claim_graph", "oracle_flow", "semantic_gating"],
            ),

            # Search retrieval (atomic steps)
            StepNode(
                step=BuildQueriesStep(),
                depends_on=["target_selection", "verify_inline_sources"],
            ),
            StepNode(
                step=WebSearchStep(
                    config=config,
                    search_mgr=self.search_mgr,
                    agent=self.agent,
                ),
                depends_on=["build_queries"],
            ),
            StepNode(
                step=RerankStep(),
                depends_on=["web_search"],
            ),
            # Fulltext fetch always included (improves quality)
            StepNode(
                step=FetchChunksStep(search_mgr=self.search_mgr),
                depends_on=["rerank_results"],
                optional=True,
            ),
            StepNode(
                step=AssembleRetrievalItemsStep(),
                depends_on=["fetch_chunks"],
            ),
            StepNode(
                step=AssertRetrievalTraceStep(),
                depends_on=["assemble_retrieval_items"],
            ),

            # Evidence collection (collect-only)
            StepNode(
                step=EvidenceCollectStep(
                    agent=self.agent,
                    search_mgr=self.search_mgr,
                    include_global_pack=True,
                ),
                depends_on=["assert_retrieval_trace"],
            ),
            StepNode(
                step=ExtractClaimsStep(
                    agent=self.agent,
                    stage="post_evidence",
                    name="enrich_claims_post_evidence",
                ),
                depends_on=["evidence_collect"],
                optional=True,
            ),

            # EVOI gating decision (after evidence collection, before expensive steps)
            # Reads EvidenceIndex to compute p_need for stance/cluster
            StepNode(
                step=EvidenceGatingStep(),
                depends_on=["enrich_claims_post_evidence"],
            ),

            # Stance annotation (controlled by EVOI gate)
            StepNode(
                step=StanceAnnotateStep(agent=self.agent),
                depends_on=["evidence_gating"],
                optional=True,
            ),

            # Clustering (controlled by EVOI gate)
            StepNode(
                step=ClusterEvidenceStep(agent=self.agent),
                depends_on=["stance_annotate"],
                optional=True,
            ),

            # Standard judging
            StepNode(
                step=JudgeStandardStep(agent=self.agent, search_mgr=self.search_mgr),
                depends_on=["cluster_evidence"],
            ),

            # Final assembly
            StepNode(
                step=AssembleStandardResultStep(),
                depends_on=["judge_standard"],
            ),

            # Cost summary
            StepNode(
                step=CostSummaryStep(),
                depends_on=["assemble_standard_result"],
            ),
            StepNode(
                step=AssertStandardResultKeysStep(),
                depends_on=["cost_summary"],
            ),
        ]

    def _build_deep_dag_nodes(self, config: Any) -> list:
        """Build DAG nodes for deep mode with per-claim judging."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps import (
            # NOTE: ClaimGraphStep NOT imported for deep mode - not needed with process_all_claims=True
            EvidenceCollectStep,
            StanceAnnotateStep,
            ClusterEvidenceStep,
            ExtractClaimsStep,
            MeteringSetupStep,
            PrepareInputStep,
            BuildQueriesStep,
            WebSearchStep,
            RerankStep,
            FetchChunksStep,
            AssembleRetrievalItemsStep,
            TargetSelectionStep,
            VerifyInlineSourcesStep,
            CostSummaryStep,
            AssertRetrievalTraceStep,
            AssertDeepJudgingStep,
        )
        from spectrue_core.pipeline.steps.deep_claim import (
            AssembleDeepResultStep,
            BuildClaimFramesStep,
            JudgeClaimsStep,
            MarkJudgeUnavailableStep,
            SummarizeEvidenceStep,
        )
        from spectrue_core.utils.trace import Trace

        # Trace: deep mode skips claim graph (all claims are processed)
        Trace.event("pipeline.deep_mode.claim_graph.disabled", {
            "reason": "process_all_claims_true",
        })

        # Get LLM client from agent
        llm_client = getattr(self.agent, "_llm", None) or getattr(self.agent, "llm", None)
        # All steps now always included (no feature flags)
        # Optional=True allows graceful failure without blocking pipeline

        return [
            # ==================== INFRASTRUCTURE ====================
            StepNode(step=MeteringSetupStep(config=config, agent=self.agent, search_mgr=self.search_mgr)),

            # ==================== INPUT PREPARATION ====================
            StepNode(
                step=PrepareInputStep(agent=self.agent, search_mgr=self.search_mgr, config=config),
                depends_on=["metering_setup"],
            ),

            # ==================== CLAIM EXTRACTION ====================
            StepNode(
                step=ExtractClaimsStep(agent=self.agent),
                depends_on=["prepare_input"],
            ),

            # Post-extraction invariant
            StepNode(
                step=AssertNonEmptyClaimsStep(),
                depends_on=["extract_claims"],
            ),

            # ==================== PARALLEL: INLINE SOURCES (NO CLAIM GRAPH IN DEEP MODE) ====================
            StepNode(
                step=VerifyInlineSourcesStep(agent=self.agent, search_mgr=self.search_mgr, config=config),
                depends_on=["extract_claims"],
            ),
            # NOTE: ClaimGraphStep removed - deep mode processes ALL claims (process_all_claims=True)
            # Graph-based pruning is unnecessary when we verify everything.

            # ==================== TARGET SELECTION (ALL CLAIMS) ====================
            StepNode(
                step=TargetSelectionStep(process_all_claims=True),
                depends_on=["extract_claims"],  # Direct dependency, no claim_graph
            ),

            # Search retrieval (atomic steps)
            StepNode(
                step=BuildQueriesStep(),
                depends_on=["target_selection", "verify_inline_sources"],
            ),
            StepNode(
                step=WebSearchStep(
                    config=config,
                    search_mgr=self.search_mgr,
                    agent=self.agent,
                ),
                depends_on=["build_queries"],
            ),
            StepNode(
                step=RerankStep(),
                depends_on=["web_search"],
            ),
            # Fulltext fetch always included
            StepNode(
                step=FetchChunksStep(search_mgr=self.search_mgr),
                depends_on=["rerank_results"],
                optional=True,
            ),
            StepNode(
                step=AssembleRetrievalItemsStep(),
                depends_on=["fetch_chunks"],
            ),
            StepNode(
                step=AssertRetrievalTraceStep(),
                depends_on=["assemble_retrieval_items"],
            ),

            # Evidence collection ONLY (NO global scoring for deep mode!)
            StepNode(
                step=EvidenceCollectStep(
                    agent=self.agent,
                    search_mgr=self.search_mgr,
                    include_global_pack=False,
                ),
                depends_on=["assert_retrieval_trace"],
            ),
            StepNode(
                step=ExtractClaimsStep(
                    agent=self.agent,
                    stage="post_evidence",
                    name="enrich_claims_post_evidence",
                ),
                depends_on=["evidence_collect"],
                optional=True,
            ),

            # Stance annotation always included
            StepNode(
                step=StanceAnnotateStep(agent=self.agent),
                depends_on=["enrich_claims_post_evidence"],
                optional=True,
            ),

            # Clustering always included
            StepNode(
                step=ClusterEvidenceStep(agent=self.agent),
                depends_on=["stance_annotate"],
                optional=True,
            ),

            # --- Per-Claim Judging (Deep Mode Only) ---

            # Build ClaimFrame for each claim
            StepNode(
                step=BuildClaimFramesStep(),
                depends_on=["cluster_evidence"],
            ),

            # Summarize evidence per claim (always included if llm_client available)
            *(
                [
                    StepNode(
                        step=SummarizeEvidenceStep(llm_client=llm_client),
                        depends_on=["build_claim_frames"],
                    )
                ]
                if llm_client
                else []
            ),

            # Judge each claim independently (per-claim RGBA)
            *(
                [
                    StepNode(
                        step=JudgeClaimsStep(llm_client=llm_client),
                        depends_on=["summarize_evidence"],
                    )
                ]
                if llm_client
                else [
                    StepNode(
                        step=MarkJudgeUnavailableStep(reason="llm_client_missing"),
                        depends_on=["build_claim_frames"],
                    )
                ]
            ),

            # Assemble deep result (per-claim verdicts, NO global RGBA)
            StepNode(
                step=AssembleDeepResultStep(),
                depends_on=["judge_claims" if llm_client else "judge_unavailable"],
            ),

            StepNode(
                step=AssertDeepJudgingStep(),
                depends_on=["assemble_deep_result"],
            ),

            StepNode(
                step=CostSummaryStep(),
                depends_on=["assert_deep_judging"],
            ),

            # NOTE: No ResultAssemblyStep fallback in deep mode!
            # Deep mode MUST return per-claim results from AssembleDeepResultStep.
            # Legacy global fields (danger_score, style_score, etc.) are NOT populated.

        ]

    def _build_extraction_dag_nodes(self, config: Any) -> list:
        """Build DAG nodes for extraction-only mode."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps import (
            ExtractClaimsStep,
            MeteringSetupStep,
            PrepareInputStep,
            ExtractionResultAssemblyStep,
        )

        return [
            # Infrastructure
            StepNode(step=MeteringSetupStep(config=config, agent=self.agent, search_mgr=self.search_mgr)),

            # Input preparation
            StepNode(
                step=PrepareInputStep(agent=self.agent, search_mgr=self.search_mgr, config=config),
                depends_on=["metering_setup"],
            ),

            # Claim extraction
            StepNode(
                step=ExtractClaimsStep(agent=self.agent),
                depends_on=["prepare_input"],
            ),
            
            # Result Assembly (Extraction specific)
            StepNode(
                step=ExtractionResultAssemblyStep(),
                depends_on=["extract_claims"],
            ),
        ]
