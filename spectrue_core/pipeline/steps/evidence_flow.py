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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


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
        from spectrue_core.verification.pipeline.pipeline_evidence import (
            EvidenceFlowInput,
            run_evidence_flow,
        )
        from spectrue_core.verification.evidence.evidence import build_evidence_pack

        try:
            claims = ctx.claims
            sources = ctx.sources
            progress_callback = ctx.get_extra("progress_callback")

            # M118/T001: Skip if valid verdict already exists (e.g. Gating Rejected or Oracle Jackpot)
            if ctx.get_extra("gating_rejected") or ctx.get_extra("oracle_hit"):
                return ctx

            # Deep mode: skip global scoring, only collect evidence
            if not self.enable_global_scoring:
                Trace.event(
                    "evidence_flow.skip_global_scoring",
                    {"reason": "deep_mode_uses_per_claim_judging", "claims_count": len(claims)},
                )
                # Build evidence pack without LLM scoring
                pack = build_evidence_pack(
                    fact=ctx.get_extra("prepared_fact", ""),
                    claims=claims,
                    sources=sources,
                    content_lang=ctx.lang,
                )

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

                # Calculate cost for deep mode
                current_cost = self.search_mgr.calculate_cost(ctx.gpt_model, ctx.search_type)

                # Create minimal result for deep mode (no global scores)
                result = {
                    "judge_mode": "deep",
                    "evidence_pack": pack,
                    "evidence_by_claim": evidence_by_claim,
                    "claim_verdicts": [],  # Will be filled by JudgeClaimsStep
                    "sources": sources,
                    "claims": claims,
                    "cost": current_cost,
                }

                Trace.event(
                    "evidence_flow.collect_only_completed",
                    {"claims_with_evidence": len(evidence_by_claim)},
                )

                return ctx.with_update(verdict=result).set_extra("evidence_by_claim", evidence_by_claim)

            # Standard mode: full evidence flow with global LLM scoring
            def _build_pack(**kwargs):
                return build_evidence_pack(
                    fact=ctx.get_extra("prepared_fact", ""),
                    claims=claims,
                    sources=kwargs.get("sources", []),
                    content_lang=ctx.lang,
                )

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
                # M119: inp.pipeline removed - mode determined by score_mode parameter
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
