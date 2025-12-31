# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Pipeline Factory

Builds Pipeline instances with the correct step composition for each mode.
This is the single place where mode -> step sequence mapping is defined.

Usage:
    from spectrue_core.pipeline.factory import PipelineFactory

    factory = PipelineFactory(search_mgr=search_mgr, agent=agent)
    pipeline = factory.build("normal")  # or "deep"
    result = await pipeline.run(ctx)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import Pipeline, Step
from spectrue_core.pipeline.mode import PipelineMode, NORMAL_MODE, DEEP_MODE, get_mode
from spectrue_core.pipeline.steps.invariants import (
    AssertNonEmptyClaimsStep,
    AssertSingleClaimStep,
    AssertSingleLanguageStep,
)
from spectrue_core.pipeline.steps.legacy import (
    LegacyPhaseRunnerStep,
    LegacyScoringStep,
    LegacyClusteringStep,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineFactory:
    """
    Factory for building Pipeline instances.

    The factory is the ONLY place where mode -> step sequence is defined.
    This eliminates scattered if-statements throughout the codebase.

    Attributes:
        search_mgr: SearchManager for retrieval steps
        agent: FactCheckerAgent for LLM steps
    """

    search_mgr: Any  # SearchManager
    agent: Any  # FactCheckerAgent

    def build(self, mode_name: str) -> Pipeline:
        """
        Build a Pipeline for the specified mode.

        Args:
            mode_name: "normal", "general", or "deep"

        Returns:
            Pipeline configured with the correct step sequence
        """
        mode = get_mode(mode_name)

        if mode.name == "normal":
            return self._build_normal_pipeline(mode)
        else:
            return self._build_deep_pipeline(mode)

    def _build_normal_pipeline(self, mode: PipelineMode) -> Pipeline:
        """
        Build pipeline for normal/general mode.

        Normal mode characteristics:
        - Single claim only
        - Single language
        - Basic search depth
        - No clustering
        """
        steps: list[Step] = [
            # Invariant gates (fail fast)
            AssertNonEmptyClaimsStep(),
            AssertSingleClaimStep(),
            AssertSingleLanguageStep(),
            # Retrieval
            LegacyPhaseRunnerStep(
                search_mgr=self.search_mgr,
                agent=self.agent,
                use_retrieval_loop=True,
            ),
            # Scoring
            LegacyScoringStep(agent=self.agent),
        ]

        logger.debug(
            "[PipelineFactory] Built normal pipeline with %d steps: %s",
            len(steps),
            [s.name for s in steps],
        )

        return Pipeline(mode=mode, steps=steps)

    def _build_deep_pipeline(self, mode: PipelineMode) -> Pipeline:
        """
        Build pipeline for deep mode.

        Deep mode characteristics:
        - Batch claims allowed
        - Multi-language allowed
        - Advanced search depth
        - Clustering enabled
        """
        steps: list[Step] = [
            # Minimal invariants (just need claims)
            AssertNonEmptyClaimsStep(),
            # Retrieval (advanced)
            LegacyPhaseRunnerStep(
                search_mgr=self.search_mgr,
                agent=self.agent,
                use_retrieval_loop=True,
            ),
            # Clustering (deep mode only)
            LegacyClusteringStep(agent=self.agent),
            # Scoring (batch)
            LegacyScoringStep(agent=self.agent),
        ]

        logger.debug(
            "[PipelineFactory] Built deep pipeline with %d steps: %s",
            len(steps),
            [s.name for s in steps],
        )

        return Pipeline(mode=mode, steps=steps)

    def build_from_profile(self, profile_name: str) -> Pipeline:
        """
        Build pipeline from a profile name.

        Maps profile names to modes:
        - "normal" -> normal mode
        - "deep" -> deep mode

        This allows integration with the existing pipeline_builder module.
        """
        mode_mapping = {
            "normal": "normal",
            "general": "normal",
            "deep": "deep",
        }

        mode_name = mode_mapping.get(profile_name.lower(), "normal")
        return self.build(mode_name)

    def build_dag(
        self,
        mode_name: str,
        *,
        config: Any,
        pipeline: Any,  # ValidationPipeline
    ) -> "DAGPipeline":
        """
        Build a DAGPipeline with decomposed steps.

        This is the M115 replacement for build() that uses native Steps
        instead of legacy wrappers.

        Args:
            mode_name: "normal", "general", or "deep"
            config: SpectrueConfig
            pipeline: ValidationPipeline (for _prepare_input access)

        Returns:
            DAGPipeline configured with decomposed steps
        """
        from spectrue_core.pipeline.dag import DAGPipeline, StepNode
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

    def _build_normal_dag_nodes(self, config: Any, pipeline: Any) -> list:
        """Build DAG nodes for normal mode."""
        from spectrue_core.pipeline.dag import StepNode
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

            # Evidence scoring
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
        """Build DAG nodes for deep mode."""
        from spectrue_core.pipeline.dag import StepNode
        from spectrue_core.pipeline.steps.decomposed import (
            MeteringSetupStep,
            PrepareInputStep,
            ExtractClaimsStep,
            ClaimGraphStep,
            TargetSelectionStep,
            SearchFlowStep,
            EvidenceFlowStep,
            ResultAssemblyStep,
        )

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

            # Evidence scoring (batch)
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
