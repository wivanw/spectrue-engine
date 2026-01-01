# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Decomposed Pipeline Steps

Native Steps extracted from ValidationPipeline.execute() monolith.
These Steps replace the 1071-line execute() method with discrete,
testable, composable units.

Migration Status:
- T001: MeteringSetupStep ✓
- T002: PrepareInputStep ✓
- T003: ExtractClaimsStep ✓
- T004-T009: TODO
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# T001: MeteringSetupStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MeteringSetupStep:
    """
    Initialize cost metering infrastructure.

    Sets up CostLedger, TavilyMeter, LLMMeter, and EmbedService metering.
    This must run before any billable operations.

    Context Input:
        - config (via extras)

    Context Output:
        - ledger (via extras)
        - tavily_meter (via extras)
        - llm_meter (via extras)
    """

    config: Any  # SpectrueConfig
    name: str = "metering_setup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Initialize metering infrastructure."""
        from spectrue_core.billing.cost_ledger import CostLedger
        from spectrue_core.billing.metering import TavilyMeter, LLMMeter
        from spectrue_core.billing.config_loader import load_pricing_policy
        from spectrue_core.billing.cost_progress import CostProgressEmitter
        from spectrue_core.utils.trace import current_trace_id
        from spectrue_core.utils.embedding_service import EmbedService

        try:
            policy = load_pricing_policy()
            ledger = CostLedger(run_id=current_trace_id())
            tavily_meter = TavilyMeter(ledger=ledger, policy=policy)
            llm_meter = LLMMeter(ledger=ledger, policy=policy)

            # Configure embedding metering
            EmbedService.configure(
                openai_api_key=getattr(self.config, "openai_api_key", None),
                meter=llm_meter,
                meter_stage="embed",
            )

            progress_emitter = CostProgressEmitter(
                ledger=ledger,
                min_delta_to_show=policy.min_delta_to_show,
                emit_cost_deltas=policy.emit_cost_deltas,
            )

            Trace.event(
                "metering_setup.completed",
                {"run_id": current_trace_id()},
            )

            return (
                ctx.set_extra("ledger", ledger)
                .set_extra("tavily_meter", tavily_meter)
                .set_extra("llm_meter", llm_meter)
                .set_extra("progress_emitter", progress_emitter)
                .set_extra("pricing_policy", policy)
            )

        except Exception as e:
            logger.exception("[MeteringSetupStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T002: PrepareInputStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PrepareInputStep:
    """
    Prepare and clean input text for processing.

    Handles:
    - URL content extraction (if input is URL)
    - Text cleaning/normalization
    - LLM-based cleaning for extension content
    - Inline source separation

    Context Input:
        - claims[0].text or first claim text as fact
        - extras: preloaded_context, preloaded_sources, source_url, needs_cleaning

    Context Output:
        - extras: prepared_fact, prepared_context, inline_sources
    """

    pipeline: Any  # ValidationPipeline (for _prepare_input method)
    name: str = "prepare_input"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Prepare input text."""
        try:
            # Get fact from claims or context
            fact = ctx.get_extra("raw_fact", "")
            if not fact and ctx.claims:
                fact = ctx.claims[0].get("text", "")

            preloaded_context = ctx.get_extra("preloaded_context")
            preloaded_sources = ctx.get_extra("preloaded_sources")
            source_url = ctx.get_extra("source_url")
            needs_cleaning = ctx.get_extra("needs_cleaning", False)
            progress_callback = ctx.get_extra("progress_callback")

            # Delegate to existing _prepare_input
            prepared = await self.pipeline._prepare_input(
                fact=fact,
                preloaded_context=preloaded_context,
                preloaded_sources=preloaded_sources,
                needs_cleaning=needs_cleaning,
                source_url=source_url,
                progress_callback=progress_callback,
            )

            Trace.event(
                "prepare_input.completed",
                {
                    "fact_len": len(prepared.fact),
                    "has_context": bool(prepared.final_context),
                    "sources_count": len(prepared.final_sources),
                    "inline_count": len(prepared.inline_sources),
                },
            )

            return (
                ctx.set_extra("prepared_fact", prepared.fact)
                .set_extra("original_fact", prepared.original_fact)
                .set_extra("prepared_context", prepared.final_context)
                .set_extra("prepared_sources", prepared.final_sources)
                .set_extra("inline_sources", prepared.inline_sources)
            )

        except Exception as e:
            logger.exception("[PrepareInputStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T003: ExtractClaimsStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractClaimsStep:
    """
    Extract verifiable claims from input text.

    Uses LLM to decompose text into atomic claims with metadata.
    Selects anchor claim for normal mode.

    Context Input:
        - extras: prepared_fact
        - lang

    Context Output:
        - claims (updated with extracted claims)
        - extras: anchor_claim_id, eligible_claims
    """

    agent: Any  # FactCheckerAgent
    name: str = "extract_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Extract claims from fact."""
        try:
            fact = ctx.get_extra("prepared_fact") or ctx.get_extra("raw_fact", "")
            progress_callback = ctx.get_extra("progress_callback")

            if progress_callback:
                await progress_callback("extracting_claims")

            # Delegate to agent skill
            extraction_result = await self.agent.claim_extraction_skill.extract(
                fact=fact,
                lang=ctx.lang,
            )

            claims = extraction_result.get("claims", [])
            if not claims:
                # Fallback: use full text as single claim
                claims = [{"id": "c1", "text": fact[:500], "importance": 1.0}]

            # Select anchor claim (highest importance)
            anchor_claim = max(claims, key=lambda c: float(c.get("importance", 0.5)))
            anchor_claim_id = str(anchor_claim.get("id", "c1"))

            # Filter eligible claims (importance >= 0.3)
            eligible_claims = [
                c for c in claims
                if float(c.get("importance", 0.5)) >= 0.3
            ]

            Trace.event(
                "extract_claims.completed",
                {
                    "total_claims": len(claims),
                    "eligible_claims": len(eligible_claims),
                    "anchor_claim_id": anchor_claim_id,
                },
            )

            return (
                ctx.with_update(claims=claims)
                .set_extra("anchor_claim_id", anchor_claim_id)
                .set_extra("eligible_claims", eligible_claims)
            )

        except Exception as e:
            logger.exception("[ExtractClaimsStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T004: ClaimGraphStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ClaimGraphStep:
    """
    Build claim graph for relationship analysis.

    Uses NetworkX to build graph of claim relationships,
    identify key claims, and compute centrality metrics.

    Context Input:
        - claims
        - extras: eligible_claims

    Context Output:
        - extras: graph_result, key_claims
    """

    config: Any  # SpectrueConfig
    name: str = "claim_graph"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Build claim graph."""
        from spectrue_core.verification.pipeline_claim_graph import run_claim_graph_flow

        try:
            eligible_claims = ctx.get_extra("eligible_claims", ctx.claims)

            # Skip if only one claim
            if len(eligible_claims) <= 1:
                Trace.event("claim_graph.skipped", {"reason": "single_claim"})
                return ctx.set_extra("graph_result", None)

            graph_result = await run_claim_graph_flow(
                claims=eligible_claims,
                config=self.config,
            )

            Trace.event(
                "claim_graph.completed",
                {
                    "claims_in_graph": len(eligible_claims),
                    "key_claims_count": len(getattr(graph_result, "key_claims", []) or []),
                },
            )

            return ctx.set_extra("graph_result", graph_result)

        except Exception as e:
            logger.warning("[ClaimGraphStep] Non-fatal failure: %s", e)
            Trace.event("claim_graph.error", {"error": str(e)})
            return ctx.set_extra("graph_result", None)


# ─────────────────────────────────────────────────────────────────────────────
# T005: TargetSelectionStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TargetSelectionStep:
    """
    Select which claims get actual searches.

    Implements the critical gate that prevents per-claim search explosion.
    Only top-K key claims trigger Tavily searches.

    Context Input:
        - extras: eligible_claims, graph_result, anchor_claim_id
        - mode (for budget_class derivation)

    Context Output:
        - extras: target_claims, deferred_claims, evidence_sharing
    """

    name: str = "target_selection"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Select verification targets."""
        from spectrue_core.verification.target_selection import select_verification_targets

        try:
            eligible_claims = ctx.get_extra("eligible_claims", ctx.claims)
            graph_result = ctx.get_extra("graph_result")
            anchor_claim_id = ctx.get_extra("anchor_claim_id")

            # Derive budget_class from mode
            budget_class = {
                "basic": "minimal",
                "advanced": "deep",
            }.get(ctx.mode.search_depth, "standard")

            # For normal mode, anchor must be in targets
            anchor_for_selection = anchor_claim_id if ctx.mode.name == "normal" else None

            # DEEP MODE: Verify ALL claims (no target selection limits)
            if ctx.mode.name == "deep":
                # All claims are targets in deep mode
                target_claims = list(eligible_claims)
                deferred_claims = []
                evidence_sharing = {}
                target_reasons = {c.get("id"): "deep_mode_all_verified" for c in target_claims}

                Trace.event(
                    "target_selection.deep_mode_all_claims",
                    {
                        "mode": ctx.mode.name,
                        "claims_count": len(target_claims),
                        "claim_ids": [c.get("id") for c in target_claims],
                    },
                )
            else:
                result = select_verification_targets(
                    claims=eligible_claims,
                    # max_targets removed: let Bayesian EVOI model compute
                    graph_result=graph_result,
                    budget_class=budget_class,
                    anchor_claim_id=anchor_for_selection,
                )
                target_claims = result.targets
                deferred_claims = result.deferred
                evidence_sharing = result.evidence_sharing
                target_reasons = result.reasons

                Trace.event(
                    "target_selection.step_completed",
                    {
                        "targets_count": len(target_claims),
                        "deferred_count": len(deferred_claims),
                    },
                )

            return (
                ctx.set_extra("target_claims", target_claims)
                .set_extra("deferred_claims", deferred_claims)
                .set_extra("evidence_sharing", evidence_sharing)
                .set_extra("target_reasons", target_reasons)
            )

        except Exception as e:
            logger.exception("[TargetSelectionStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T006: SearchFlowStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchFlowStep:
    """
    Execute search retrieval for target claims.

    Wraps run_search_flow() to collect sources from Tavily/CSE.

    Context Input:
        - extras: target_claims, inline_sources
        - lang, search_type, gpt_model

    Context Output:
        - sources (updated with collected sources)
        - extras: execution_state
    """

    config: Any  # SpectrueConfig
    search_mgr: Any  # SearchManager
    agent: Any  # FactCheckerAgent
    name: str = "search_flow"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute search retrieval."""
        from spectrue_core.verification.pipeline_search import (
            SearchFlowInput,
            SearchFlowState,
            run_search_flow,
        )

        try:
            target_claims = ctx.get_extra("target_claims", ctx.claims)
            inline_sources = ctx.get_extra("inline_sources", [])
            progress_callback = ctx.get_extra("progress_callback")


            def can_add_search(model: str, search_type: str, max_cost: int | None) -> bool:
                # Simple budget check
                return True

            inp = SearchFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                max_cost=ctx.get_extra("max_cost"),
                article_intent=ctx.get_extra("article_intent", "general"),
                search_queries=ctx.get_extra("search_queries", []),
                claims=target_claims,
                preloaded_context=ctx.get_extra("prepared_context"),
                progress_callback=progress_callback,
                inline_sources=inline_sources,
                pipeline=ctx.mode.name,
            )

            state = SearchFlowState(
                final_context="",
                final_sources=[],
                preloaded_context=ctx.get_extra("prepared_context"),
                used_orchestration=False,
            )

            result_state = await run_search_flow(
                config=self.config,
                search_mgr=self.search_mgr,
                agent=self.agent,
                can_add_search=can_add_search,
                inp=inp,
                state=state,
            )

            Trace.event(
                "search_flow.step_completed",
                {
                    "sources_collected": len(result_state.final_sources),
                    "used_orchestration": result_state.used_orchestration,
                },
            )

            return (
                ctx.with_update(sources=result_state.final_sources)
                .set_extra("search_context", result_state.final_context)
                .set_extra("execution_state", result_state.execution_state)
            )

        except Exception as e:
            logger.exception("[SearchFlowStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T007: EvidenceFlowStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvidenceFlowStep:
    """
    Process evidence and compute verdicts.

    Orchestrates stance clustering, evidence scoring, and
    RGBA computation for all claims.
    
    Args:
        enable_global_scoring: If False, skip the global LLM scoring call.
            Deep mode sets this to False because it uses per-claim JudgeClaimsStep.
    """

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    enable_global_scoring: bool = True  # Standard mode: True, Deep mode: False
    name: str = "evidence_flow"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Process evidence and compute verdicts."""
        from spectrue_core.verification.pipeline_evidence import (
            EvidenceFlowInput,
            run_evidence_flow,
        )
        from spectrue_core.verification.evidence import build_evidence_pack

        try:
            claims = ctx.claims
            sources = ctx.sources
            progress_callback = ctx.get_extra("progress_callback")

            # Deep mode: skip global scoring, only collect evidence
            if not self.enable_global_scoring:
                Trace.event(
                    "evidence_flow.skip_global_scoring",
                    {"reason": "deep_mode_uses_per_claim_judging", "claims_count": len(claims)},
                )
                # Build evidence pack without LLM scoring
                pack = build_evidence_pack(sources)

                # Prepare evidence_by_claim for BuildClaimFramesStep
                evidence_by_claim: dict = {}
                for src in sources:
                    if not isinstance(src, dict):
                        continue
                    cid = src.get("claim_id")
                    if cid:
                        if cid not in evidence_by_claim:
                            evidence_by_claim[cid] = []
                        evidence_by_claim[cid].append(src)

                # Create minimal result for deep mode (no global scores)
                result = {
                    "judge_mode": "deep",
                    "evidence_pack": pack,
                    "evidence_by_claim": evidence_by_claim,
                    "claim_verdicts": [],  # Will be filled by JudgeClaimsStep
                    "sources": sources,
                    "claims": claims,
                }

                Trace.event(
                    "evidence_flow.collect_only_completed",
                    {"claims_with_evidence": len(evidence_by_claim)},
                )

                return ctx.with_update(verdict=result).set_extra("evidence_by_claim", evidence_by_claim)

            # Standard mode: full evidence flow with global LLM scoring
            def _build_pack(**kwargs):
                return build_evidence_pack(kwargs.get("sources", []))

            def _noop_enrich(sources):
                return sources

            inp = EvidenceFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                original_fact=ctx.get_extra("original_fact", ""),
                lang=ctx.lang,
                content_lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                progress_callback=progress_callback,
                pipeline=ctx.mode.name,
            )

            result = await run_evidence_flow(
                agent=self.agent,
                search_mgr=self.search_mgr,
                build_evidence_pack=_build_pack,
                enrich_sources_with_trust=_noop_enrich,
                calibration_registry=None,
                inp=inp,
                claims=claims,
                sources=sources,
                score_mode="standard",  # M118: explicit mode (never "parallel" here, deep uses branch above)
            )


            Trace.event(
                "evidence_flow.step_completed",
                {"verified_score": result.get("verified_score", 0.5)},
            )

            return ctx.with_update(verdict=result).set_extra("rgba", result.get("rgba"))

        except Exception as e:
            logger.exception("[EvidenceFlowStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


# ─────────────────────────────────────────────────────────────────────────────
# T008: OracleFlowStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OracleFlowStep:
    """
    Check authoritative sources (Oracle) before search.

    Fast path: If fact has known status in Oracle, skip search.
    """

    pipeline: Any  # ValidationPipeline
    name: str = "oracle_flow"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Check Oracle for known status."""
        from spectrue_core.verification.pipeline_oracle import OracleFlowInput, run_oracle_flow

        try:
            fact = ctx.get_extra("prepared_fact", "")
            article_intent = ctx.get_extra("article_intent", "general")

            # Skip Oracle for opinion/prediction
            if article_intent in {"opinion", "prediction"}:
                Trace.event("oracle_flow.skipped", {"reason": "intent_skip"})
                return ctx.set_extra("oracle_hit", False)

            inp = OracleFlowInput(
                fact=fact,
                lang=ctx.lang,
                progress_callback=ctx.get_extra("progress_callback"),
            )

            result = await run_oracle_flow(pipeline=self.pipeline, inp=inp)
            oracle_hit = result.get("hit", False)

            Trace.event("oracle_flow.step_completed", {"hit": oracle_hit})

            return ctx.set_extra("oracle_hit", oracle_hit).set_extra(
                "oracle_result", result if oracle_hit else None
            )

        except Exception as e:
            logger.warning("[OracleFlowStep] Non-fatal: %s", e)
            return ctx.set_extra("oracle_hit", False)


# ─────────────────────────────────────────────────────────────────────────────
# T009: ResultAssemblyStep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResultAssemblyStep:
    """Assemble final result payload."""

    name: str = "result_assembly"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Assemble final result."""
        try:
            verdict = ctx.verdict or {}
            sources = ctx.sources or []
            ledger = ctx.get_extra("ledger")

            cost_summary = ledger.to_summary_dict() if ledger else None

            final_result = {
                "status": "ok",
                "verified_score": verdict.get("verified_score", 0.5),
                "explainability_score": verdict.get("explainability_score", 0.0),
                "danger_score": verdict.get("danger_score", 0.0),
                "rgba": ctx.get_extra("rgba", [0.0, 0.0, 0.0, 0.5]),
                "sources": sources,
                "rationale": verdict.get("rationale", ""),
                "cost_summary": cost_summary,
            }

            if ctx.get_extra("oracle_hit"):
                final_result["oracle_hit"] = True

            Trace.event("result_assembly.completed", {"verified_score": final_result["verified_score"]})

            return ctx.set_extra("final_result", final_result)

        except Exception as e:
            logger.exception("[ResultAssemblyStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
