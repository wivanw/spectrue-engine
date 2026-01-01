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
    AssertSingleClaimStep,
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

    def build(
        self,
        mode_name: str,
        *,
        config: Any,
        pipeline: Any,  # ValidationPipeline (for _prepare_input access)
    ) -> "DAGPipeline":
        """
        Build a DAGPipeline with decomposed steps.

        This is the primary entry point for pipeline construction.
        All mode logic is encapsulated here.

        Args:
            mode_name: "normal", "general", or "deep"
            config: SpectrueConfig
            pipeline: ValidationPipeline (for _prepare_input access)

        Returns:
            DAGPipeline configured with decomposed steps for the mode
        """
        from spectrue_core.pipeline.dag import DAGPipeline

        mode = get_mode(mode_name)

        if mode.name == "normal":
            nodes = self._build_normal_dag_nodes(config, pipeline)
        else:
            nodes = self._build_deep_dag_nodes(config, pipeline)

        logger.debug(
            "[PipelineFactory] Built DAG pipeline with %d nodes: %s",
            len(nodes),
            [n.name for n in nodes],
        )

        return DAGPipeline(mode=mode, nodes=nodes)

    def build_from_profile(self, profile_name: str, *, config: Any, pipeline: Any) -> "DAGPipeline":
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
        return self.build(mode_name, config=config, pipeline=pipeline)

    def _build_normal_dag_nodes(self, config: Any, pipeline: Any) -> list:
        """Build DAG nodes for normal mode."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps.decomposed import (
            ClaimGraphStep,
            EvidenceFlowStep,
            ExtractClaimsStep,
            MeteringSetupStep,
            OracleFlowStep,
            PrepareInputStep,
            ResultAssemblyStep,
            SearchFlowStep,
            TargetSelectionStep,
        )

        return [
            # Infrastructure
            StepNode(step=MeteringSetupStep(config=config)),

            # Invariants
            StepNode(
                step=AssertNonEmptyClaimsStep(),
                depends_on=["metering_setup"],
            ),

            # Input preparation
            StepNode(
                step=PrepareInputStep(pipeline=pipeline),
                depends_on=["assert_non_empty_claims"],
            ),

            # Claim extraction
            StepNode(
                step=ExtractClaimsStep(agent=self.agent),
                depends_on=["prepare_input"],
            ),

            # Normal mode: single claim check
            StepNode(
                step=AssertSingleClaimStep(),
                depends_on=["extract_claims"],
            ),

            # Oracle fast path (optional)
            StepNode(
                step=OracleFlowStep(pipeline=pipeline),
                depends_on=["assert_single_claim"],
                optional=True,
            ),

            # Claim graph (optional, parallel with oracle)
            StepNode(
                step=ClaimGraphStep(config=config),
                depends_on=["extract_claims"],
                optional=True,
            ),

            # Target selection
            StepNode(
                step=TargetSelectionStep(),
                depends_on=["claim_graph", "oracle_flow"],
            ),

            # Search retrieval
            StepNode(
                step=SearchFlowStep(
                    config=config,
                    search_mgr=self.search_mgr,
                    agent=self.agent,
                ),
                depends_on=["target_selection"],
            ),

            # Evidence scoring (enable_global_scoring=True is default)
            StepNode(
                step=EvidenceFlowStep(agent=self.agent, search_mgr=self.search_mgr),
                depends_on=["search_flow"],
            ),

            # Final assembly
            StepNode(
                step=ResultAssemblyStep(),
                depends_on=["evidence_flow"],
            ),
        ]

    def _build_deep_dag_nodes(self, config: Any, pipeline: Any) -> list:
        """Build DAG nodes for deep mode with per-claim judging."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps.decomposed import (
            ClaimGraphStep,
            EvidenceFlowStep,
            ExtractClaimsStep,
            MeteringSetupStep,
            PrepareInputStep,
            SearchFlowStep,
            TargetSelectionStep,
        )
        from spectrue_core.pipeline.steps.deep_claim import (
            AssembleDeepResultStep,
            BuildClaimFramesStep,
            JudgeClaimsStep,
            SummarizeEvidenceStep,
        )

        # Get LLM client from agent
        llm_client = getattr(self.agent, "_llm", None) or getattr(self.agent, "llm", None)

        return [
            # Infrastructure
            StepNode(step=MeteringSetupStep(config=config)),

            # Minimal invariants
            StepNode(
                step=AssertNonEmptyClaimsStep(),
                depends_on=["metering_setup"],
            ),

            # Input preparation
            StepNode(
                step=PrepareInputStep(pipeline=pipeline),
                depends_on=["assert_non_empty_claims"],
            ),

            # Claim extraction
            StepNode(
                step=ExtractClaimsStep(agent=self.agent),
                depends_on=["prepare_input"],
            ),

            # Claim graph (for clustering)
            StepNode(
                step=ClaimGraphStep(config=config),
                depends_on=["extract_claims"],
            ),

            # Target selection (more targets for deep)
            StepNode(
                step=TargetSelectionStep(),
                depends_on=["claim_graph"],
            ),

            # Search retrieval (advanced depth)
            StepNode(
                step=SearchFlowStep(
                    config=config,
                    search_mgr=self.search_mgr,
                    agent=self.agent,
                ),
                depends_on=["target_selection"],
            ),

            # Evidence collection ONLY (NO global scoring for deep mode!)
            # Deep mode uses per-claim JudgeClaimsStep instead of batch scoring
            StepNode(
                step=EvidenceFlowStep(
                    agent=self.agent,
                    search_mgr=self.search_mgr,
                    enable_global_scoring=False,  # <-- KEY: No batch LLM call
                ),
                depends_on=["search_flow"],
            ),

            # --- Per-Claim Judging (Deep Mode Only) ---

            # Build ClaimFrame for each claim
            StepNode(
                step=BuildClaimFramesStep(),
                depends_on=["evidence_flow"],
            ),

            # Summarize evidence per claim
            StepNode(
                step=SummarizeEvidenceStep(llm_client=llm_client) if llm_client else SummarizeEvidenceStep.__new__(SummarizeEvidenceStep),
                depends_on=["build_claim_frames"],
                optional=llm_client is None,
            ),

            # Judge each claim independently (per-claim RGBA)
            StepNode(
                step=JudgeClaimsStep(llm_client=llm_client) if llm_client else JudgeClaimsStep.__new__(JudgeClaimsStep),
                depends_on=["summarize_evidence"],
                optional=llm_client is None,
            ),

            # Assemble deep result (per-claim verdicts, NO global RGBA)
            StepNode(
                step=AssembleDeepResultStep(),
                depends_on=["judge_claims"],
            ),

            # NOTE: No ResultAssemblyStep fallback in deep mode!
            # Deep mode MUST return per-claim results from AssembleDeepResultStep.
            # Legacy global fields (danger_score, style_score, etc.) are NOT populated.
        ]

    # Deprecated alias for backward compatibility
    def build_dag(
        self,
        mode_name: str,
        *,
        config: Any,
        pipeline: Any,
    ) -> "DAGPipeline":
        """
        Deprecated: Use build() instead.

        This method exists for backward compatibility during migration.
        """
        logger.warning(
            "[PipelineFactory] build_dag() is deprecated, use build() instead"
        )
        return self.build(mode_name, config=config, pipeline=pipeline)
