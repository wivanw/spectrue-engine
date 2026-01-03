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

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ResultAssemblyStep:
    """Assemble final result payload."""

    name: str = "result_assembly"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Assemble final result."""
        try:
            verdict = ctx.verdict or {}
            # Sources: prefer verdict sources (often enriched), fallback to ctx.sources
            sources = verdict.get("sources") or ctx.sources or []
            ledger = ctx.get_extra("ledger")

            cost_summary = ledger.to_summary_dict() if ledger else None

            final_result = {
                "status": "ok",
                "verified_score": verdict.get("verified_score", 0.0),
                "explainability_score": verdict.get("explainability_score", 0.0),
                "danger_score": verdict.get("danger_score", 0.0),
                "style_score": verdict.get("style_score", 0.0),
                "bias_score": verdict.get("style_score", 0.0), # Compat alias
                "rgba": ctx.get_extra("rgba", [0.0, 0.0, 0.0, 0.5]),
                "sources": sources,
                "rationale": verdict.get("rationale", ""),
                "analysis": verdict.get("analysis") or verdict.get("rationale", ""), # Legacy compat
                "cost_summary": cost_summary,
                "cost": verdict.get("cost", 0.0),
            }

            # --- Legacy/UI compatibility ---
            # Older front-end expects:
            #   - `text` : the main displayed text
            #   - `details` : list of claim strings shown as chips
            #   - `_extracted_claims` : raw extracted claims (debug/optional)
            #   - `claim_verdicts` : per-claim scoring results
            # Newer pipeline stores:
            #   - `fact` / `original_fact` / `prepared_fact`
            #   - `claim_verdicts` (in verdict)
            # Provide legacy aliases to avoid breaking standard mode UI.

            # Text: prefer prepared_fact (extracted article text), fallback to fact/original_fact/input
            prepared_fact = ctx.get_extra("prepared_fact") or ""
            original_fact = ctx.get_extra("original_fact") or ctx.get_extra("fact") or ""
            final_result["text"] = prepared_fact or original_fact or ctx.get_extra("input_text") or ""
            final_result["fact"] = original_fact  # Keep for any code expecting this
            final_result["original_fact"] = original_fact

            # Details: list of claim text strings for the UI chips
            claim_verdicts = verdict.get("claim_verdicts") or []

            # Build mapping from extracted claims (ctx.claims) to enrich UI fields
            claim_text_by_id: dict[str, str] = {}
            for c in (ctx.claims or []):
                if not isinstance(c, dict):
                    continue
                cid = c.get("id") or c.get("claim_id")
                if not cid:
                    continue
                txt = (c.get("normalized_text") or c.get("text") or c.get("claim_text") or "").strip()
                if txt:
                    claim_text_by_id[str(cid)] = txt

            # Enrich verdict claim entries with text if LLM didn't echo it (does NOT modify scores)
            enriched_cvs = []
            for cv in claim_verdicts:
                if not isinstance(cv, dict):
                    continue
                cid = str(cv.get("claim_id") or cv.get("id") or "")
                if cid and not (cv.get("text") or cv.get("claim_text")):
                    fallback_txt = claim_text_by_id.get(cid)
                    if fallback_txt:
                        cv = dict(cv)
                        cv["text"] = fallback_txt
                enriched_cvs.append(cv)

            final_result["claim_verdicts"] = enriched_cvs

            # Details: prefer enriched verdicts; fallback to extracted claims list
            details = [
                (cv.get("text") or cv.get("claim_text") or cv.get("claim") or "").strip()
                for cv in enriched_cvs
                if isinstance(cv, dict) and (cv.get("text") or cv.get("claim_text") or cv.get("claim"))
            ]
            if not details:
                details = list(claim_text_by_id.values())
            final_result["details"] = details

            # Extracted claims: raw claims list for debugging
            final_result["_extracted_claims"] = ctx.claims or []

            # Anchor claim:
            # - prefer verdict.anchor_claim if present
            # - else ctx.extras.anchor_claim
            # - else first extracted claim as a deterministic UI anchor (no scoring impact)
            anchor = verdict.get("anchor_claim") or ctx.get_extra("anchor_claim")
            if not anchor and (ctx.claims or []):
                c0 = ctx.claims[0]
                if isinstance(c0, dict):
                    anchor = {
                        "id": c0.get("id") or c0.get("claim_id") or "c1",
                        "text": (c0.get("normalized_text") or c0.get("text") or "").strip(),
                    }
            if anchor:
                final_result["anchor_claim"] = anchor

            if not ledger:
                logger.warning(f"[ResultAssemblyStep] Ledger not found in context. Cost summary will be empty/zero. Extras keys: {list(ctx.extras.keys())}")

            if ctx.get_extra("oracle_hit"):
                final_result["oracle_hit"] = True

            # Debug trace: log all keys being sent to frontend
            Trace.event("final_result.keys", {"keys": sorted(final_result.keys())})
            Trace.event("result_assembly.completed", {"verified_score": final_result["verified_score"]})

            return ctx.set_extra("final_result", final_result)


        except Exception as e:
            logger.exception("[ResultAssemblyStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
