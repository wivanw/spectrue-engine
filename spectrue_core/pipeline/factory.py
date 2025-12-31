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
