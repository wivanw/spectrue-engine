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
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class OracleFlowStep:
    """
    Check authoritative sources (Oracle) before search.

    Fast path: If fact has known status in Oracle, skip search.
    """

    search_mgr: Any # SearchManager
    name: str = "oracle_flow"

    @staticmethod
    def _create_evidence_source(oracle_result: dict) -> dict:
        """Convert Oracle result to evidence source."""
        return {
            "url": oracle_result.get("url"),
            "title": oracle_result.get("title", "Oracle Verification"),
            "content": oracle_result.get("snippet", ""),
            "source_type": "official",
            "evidence_tier": "A",
            "is_trusted": True,
            "relevance_score": oracle_result.get("relevance_score", 0.0),
            "stance": "SUPPORT" if oracle_result.get("relevance_score", 0.0) > 0.7 else "REFUTE", # Simple heuristic
            "publisher": oracle_result.get("publisher"),
        }

    @staticmethod
    async def _finalize_jackpot(oracle_result: dict) -> dict:
        """
        Build final result from Jackpot (definitive match).
        
        If status is REFUTED/FALSE, verified_score should be low.
        If status is VERIFIED/TRUE, verified_score should be high.
        relevance_score is typically 'similarity to fact', so high relevance + REFUTED = low veracity.
        """
        relevance = oracle_result.get("relevance_score", 0.95)
        status = str(oracle_result.get("status", "")).upper()
        
        verified_score = relevance
        if status in ("REFUTED", "FALSE", "MISLEADING", "INCORRECT", "FAKE"):
            verified_score = 1.0 - relevance
            if verified_score < 0:
                verified_score = 0.0
            
        return {
            "verified_score": verified_score,
            "rationale": f"Authoritative Check ({status}): {oracle_result.get('publisher', 'Oracle')} - {oracle_result.get('summary', '')}",
            "rgba": [0.0, 0.0, 0.0, relevance], 
            "sources": [{
                "url": oracle_result.get("url"),
                "title": oracle_result.get("title"),
                "is_jackpot": True,
                "stance": "REFUTE" if status in ("REFUTED", "FALSE") else "SUPPORT"
            }],
            "status": "jackpot",
            "oracle_hit": True
        }

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Check Oracle for known status."""
        from spectrue_core.verification.pipeline.pipeline_oracle import (
            OracleFlowInput, run_oracle_flow,
            ORACLE_CHECK_INTENT, ORACLE_SKIP_INTENT
        )

        try:
            fact = ctx.get_extra("prepared_fact", "")
            article_intent = ctx.get_extra("article_intent", "general")

            # Legacy logic: check if LLM flagged it OR intent requires it
            check_oracle = ctx.get_extra("check_oracle", True) # Default to True to be safe? Or False?
            
            should_run = check_oracle
            if not should_run:
                # Fallback to intent check
                if article_intent not in {"opinion", "prediction", "analysis"}:
                    should_run = True
            
            if not should_run:
                Trace.event("oracle_flow.skipped", {"reason": "check_oracle_false_and_intent_skip"})
                return ctx.set_extra("oracle_hit", False)

            inp = OracleFlowInput(
                original_fact=ctx.get_extra("raw_fact", fact),
                fast_query=ctx.get_extra("fast_query", fact), # Fallback to fact if missing
                lang=ctx.lang,
                article_intent=article_intent,
                should_check_oracle=check_oracle,
                claims=ctx.get_extra("eligible_claims", ctx.claims), # Use eligible or all
                oracle_check_intents=ORACLE_CHECK_INTENT,
                oracle_skip_intents=ORACLE_SKIP_INTENT,
                progress_callback=ctx.get_extra("progress_callback"),
            )

            result = await run_oracle_flow(
                search_mgr=self.search_mgr,
                inp=inp,
                finalize_jackpot=self._finalize_jackpot,
                create_evidence_source=self._create_evidence_source
            )
            
            # Handle dataclass result
            # OracleFlowResult(early_result, evidence_source, ran_oracle, skip_reason)
            oracle_hit = False
            early_result = getattr(result, "early_result", None)
            
            if early_result:
                # Jackpot!
                oracle_hit = True
                Trace.event("oracle_flow.jackpot", {"score": early_result.get("verified_score")})
                
                # Update context with final result and stop search
                ctx = ctx.with_update(
                    verdict=early_result,
                    sources=early_result.get("sources", [])
                )
                
                # Clear candidates to stop SearchFlow
                ctx = ctx.set_extra("search_candidates", [])
                
                # Set final result for ResultAssembly
                ctx = ctx.set_extra("final_result", early_result).set_extra("rgba", early_result.get("rgba"))
                
            elif getattr(result, "evidence_source", None):
                 # Partial evidence finding
                 oracle_hit = True
                 # Add to sources? EvidenceFlow handles this usually if we pass sources?
                 pass

            Trace.event("oracle_flow.step_completed", {"hit": oracle_hit})

            return ctx.set_extra("oracle_hit", oracle_hit).set_extra(
                "oracle_result", result if oracle_hit else None
            )

        except Exception as e:
            logger.warning("[OracleFlowStep] Non-fatal: %s", e)
            return ctx.set_extra("oracle_hit", False)
