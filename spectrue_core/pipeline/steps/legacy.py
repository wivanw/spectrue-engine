# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Legacy Steps

Wrappers around existing monolithic components to enable incremental migration.
These steps delegate to legacy implementations while conforming to the Step protocol.

Migration Strategy:
1. Wrap legacy component in a Step
2. Replace call sites with pipeline.run()
3. Gradually replace legacy internals with native Steps
4. Eventually remove legacy wrappers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError

if TYPE_CHECKING:
    from spectrue_core.verification.phase_runner import PhaseRunner
    from spectrue_core.verification.execution_plan import ExecutionPlan
    from spectrue_core.verification.search_mgr import SearchManager
    from spectrue_core.agents.fact_checker_agent import FactCheckerAgent


logger = logging.getLogger(__name__)


@dataclass
class LegacyPhaseRunnerStep:
    """
    Wraps existing PhaseRunner for gradual migration.

    This step delegates to the existing PhaseRunner.run_all_claims()
    method, preserving current behavior while conforming to Step protocol.

    After sufficient testing, this wrapper can be replaced with
    native pipeline steps.

    Attributes:
        search_mgr: SearchManager instance for searches
        agent: FactCheckerAgent for LLM operations
        use_retrieval_loop: Whether to use iterative retrieval
        max_concurrent: Max parallel searches
    """

    search_mgr: Any  # SearchManager
    agent: Any | None = None  # FactCheckerAgent
    use_retrieval_loop: bool = True
    max_concurrent: int = 3
    ev_stop_params: Any | None = None
    name: str = "legacy_phase_runner"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute PhaseRunner and collect sources."""
        from spectrue_core.verification.phase_runner import PhaseRunner
        from spectrue_core.verification.execution_plan import ExecutionPlan, Phase, BudgetClass
        from spectrue_core.utils.trace import Trace

        claims = ctx.claims
        if not claims:
            logger.warning("[LegacyPhaseRunnerStep] No claims to process")
            return ctx

        # Build minimal execution plan from claims
        execution_plan = self._build_execution_plan(ctx)

        # Convert dict claims to Claim type expected by PhaseRunner
        claim_list = [self._to_claim_type(c) for c in claims]

        # Create PhaseRunner with config from context
        runner = PhaseRunner(
            search_mgr=self.search_mgr,
            max_concurrent=self.max_concurrent,
            progress_callback=ctx.get_extra("progress_callback"),
            use_retrieval_loop=self.use_retrieval_loop,
            gpt_model=ctx.gpt_model,
            search_type=ctx.search_type,
            max_cost=ctx.get_extra("max_cost"),
            inline_sources=ctx.get_extra("inline_sources", []),
            agent=self.agent,
            ev_stop_params=self.ev_stop_params,
        )

        try:
            # Run retrieval for all claims
            evidence_map = await runner.run_all_claims(
                claims=claim_list,
                execution_plan=execution_plan,
            )

            # Flatten sources from all claims
            all_sources: list[dict] = []
            for claim_id, sources in evidence_map.items():
                all_sources.extend(sources)

            # Trace result
            Trace.event(
                "legacy_phase_runner.completed",
                {
                    "claims_processed": len(claims),
                    "sources_collected": len(all_sources),
                    "use_retrieval_loop": self.use_retrieval_loop,
                },
            )

            # Update context with collected sources
            return ctx.with_update(
                sources=list(ctx.sources) + all_sources,
            ).set_extra("evidence_map", evidence_map).set_extra(
                "execution_state", runner.execution_state
            )

        except Exception as e:
            logger.exception("[LegacyPhaseRunnerStep] Retrieval failed: %s", e)
            Trace.event(
                "legacy_phase_runner.error",
                {"error": str(e), "error_type": type(e).__name__},
            )
            raise PipelineExecutionError(self.name, str(e), cause=e) from e

    def _build_execution_plan(self, ctx: PipelineContext) -> "ExecutionPlan":
        """Build ExecutionPlan from context."""
        from spectrue_core.verification.execution_plan import (
            ExecutionPlan,
            Phase,
            BudgetClass,
        )
        from spectrue_core.schema.claim_metadata import EvidenceChannel

        # Determine budget class from mode
        if ctx.mode.name == "deep":
            budget_class = BudgetClass.DEEP
            phases_enabled = ["A", "B", "C", "D"]
            max_results = 10
        else:
            budget_class = BudgetClass.STANDARD
            phases_enabled = ["A", "B"]
            max_results = 5

        # Build phases
        phases = []
        for phase_id in phases_enabled:
            phase = Phase(
                phase_id=phase_id,
                channels=[EvidenceChannel.TRUSTED_MEDIA],
                search_depth=ctx.mode.search_depth,
                locale=ctx.lang,
                max_results=max_results,
            )
            phases.append(phase)

        # Build per-claim phase map
        claim_phases = {}
        for claim in ctx.claims:
            claim_id = claim.get("id") or claim.get("claim_id") or "c1"
            claim_phases[str(claim_id)] = phases

        return ExecutionPlan(
            budget_class=budget_class,
            claim_phases=claim_phases,
        )

    def _to_claim_type(self, claim_dict: dict) -> dict:
        """Ensure claim has required fields for PhaseRunner."""
        # PhaseRunner expects dict with at minimum 'id' and 'text'
        if "id" not in claim_dict and "claim_id" not in claim_dict:
            claim_dict = {**claim_dict, "id": "c1"}
        return claim_dict


@dataclass
class LegacyScoringStep:
    """
    Wraps existing scoring logic for gradual migration.

    This step delegates to the agent's scoring skill,
    preserving current RGBA computation behavior.

    Attributes:
        agent: FactCheckerAgent for scoring
    """

    agent: Any  # FactCheckerAgent
    name: str = "legacy_scoring"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute scoring and compute verdict."""
        from spectrue_core.utils.trace import Trace
        from spectrue_core.verification.evidence import build_evidence_pack
        from spectrue_core.verification.scoring_aggregation import aggregate_claim_verdict

        claims = ctx.claims
        sources = ctx.sources

        if not claims:
            logger.warning("[LegacyScoringStep] No claims to score")
            return ctx.with_update(verdict={"verified_score": 0.5})

        try:
            # Build evidence pack
            evidence_pack = build_evidence_pack(sources)

            # For single claim (normal mode), score directly
            if len(claims) == 1:
                claim = claims[0]
                verdict = await self._score_single_claim(claim, evidence_pack, ctx)
            else:
                # For batch (deep mode), aggregate verdicts
                verdict = await self._score_batch_claims(claims, evidence_pack, ctx)

            Trace.event(
                "legacy_scoring.completed",
                {
                    "claims_scored": len(claims),
                    "verified_score": verdict.get("verified_score", 0.5),
                },
            )

            return ctx.with_update(verdict=verdict, evidence=evidence_pack)

        except Exception as e:
            logger.exception("[LegacyScoringStep] Scoring failed: %s", e)
            Trace.event(
                "legacy_scoring.error",
                {"error": str(e), "error_type": type(e).__name__},
            )
            raise PipelineExecutionError(self.name, str(e), cause=e) from e

    async def _score_single_claim(
        self, claim: dict, evidence_pack: dict, ctx: PipelineContext
    ) -> dict:
        """Score a single claim."""
        # Delegate to agent scoring
        result = await self.agent.scoring_skill.analyze(
            fact=claim.get("text", ""),
            context=str(evidence_pack),
            gpt_model=ctx.gpt_model,
            lang=ctx.lang,
            analysis_mode=ctx.search_type,
        )
        return result

    async def _score_batch_claims(
        self, claims: list[dict], evidence_pack: dict, ctx: PipelineContext
    ) -> dict:
        """Score multiple claims and aggregate."""
        verdicts = []
        for claim in claims:
            result = await self._score_single_claim(claim, evidence_pack, ctx)
            verdicts.append(result)

        # Simple aggregation: weighted mean of verified_scores
        if not verdicts:
            return {"verified_score": 0.5}

        total_score = sum(v.get("verified_score", 0.5) for v in verdicts)
        avg_score = total_score / len(verdicts)

        return {
            "verified_score": avg_score,
            "claim_verdicts": verdicts,
            "aggregation": "weighted_mean",
        }


@dataclass
class LegacyClusteringStep:
    """
    Wraps existing stance clustering for gradual migration.

    Only activated in deep mode where clustering is enabled.

    Attributes:
        agent: FactCheckerAgent for clustering skill
    """

    agent: Any  # FactCheckerAgent
    name: str = "legacy_clustering"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute stance clustering if enabled."""
        from spectrue_core.utils.trace import Trace

        # Skip if clustering not allowed for this mode
        if not ctx.mode.allow_clustering:
            Trace.event("legacy_clustering.skipped", {"reason": "mode_disallows"})
            return ctx

        sources = ctx.sources
        if not sources:
            Trace.event("legacy_clustering.skipped", {"reason": "no_sources"})
            return ctx

        try:
            # Delegate to clustering skill
            clustering_result = await self.agent.clustering_skill.cluster_stances(
                sources=sources,
                claims=ctx.claims,
            )

            Trace.event(
                "legacy_clustering.completed",
                {
                    "sources_clustered": len(sources),
                    "clusters_found": len(clustering_result.get("clusters", [])),
                },
            )

            return ctx.set_extra("clustering_result", clustering_result)

        except Exception as e:
            logger.warning("[LegacyClusteringStep] Clustering failed: %s", e)
            Trace.event(
                "legacy_clustering.error",
                {"error": str(e), "error_type": type(e).__name__},
            )
            # Non-fatal: continue without clustering
            return ctx
