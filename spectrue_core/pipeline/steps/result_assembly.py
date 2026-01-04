# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from __future__ import annotations

import logging
from dataclasses import dataclass

from spectrue_core.pipeline.contracts import CLAIMS_KEY, INPUT_DOC_KEY, Claims, InputDoc
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def _build_claim_text_lookup(claims_contract: Claims | None) -> dict[str, str]:
    if claims_contract is None:
        return {}
    lookup: dict[str, str] = {}
    for claim in claims_contract.claims:
        if claim.text:
            lookup[str(claim.id)] = claim.text
    return lookup


def _build_anchor_claim(
    verdict: dict,
    claims_contract: Claims | None,
    claim_text_by_id: dict[str, str],
    fallback_claims: list[dict],
) -> dict | None:
    anchor = verdict.get("anchor_claim")
    if anchor:
        return anchor

    anchor_id = claims_contract.anchor_claim_id if claims_contract else None
    if anchor_id:
        return {
            "id": anchor_id,
            "text": claim_text_by_id.get(anchor_id, ""),
        }

    if fallback_claims:
        c0 = fallback_claims[0]
        if isinstance(c0, dict):
            return {
                "id": c0.get("id") or c0.get("claim_id") or "c1",
                "text": (c0.get("normalized_text") or c0.get("text") or "").strip(),
            }

    return None


@dataclass
class AssembleStandardResultStep:
    """Assemble standard-mode payload by mapping prior step outputs."""

    name: str = "assemble_standard_result"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            verdict = ctx.verdict or {}
            sources = verdict.get("sources") or ctx.sources or []
            analysis_mode = "deep" if ctx.mode.name == "deep" else "general"

            input_doc = ctx.get_extra(INPUT_DOC_KEY)
            if not isinstance(input_doc, InputDoc):
                input_doc = None

            claims_contract = ctx.get_extra(CLAIMS_KEY)
            if not isinstance(claims_contract, Claims):
                claims_contract = None

            prepared_text = (
                input_doc.prepared_text
                if input_doc
                else ctx.get_extra("prepared_fact") or ""
            )
            original_fact = ctx.get_extra("original_fact") or ctx.get_extra("fact") or ""

            rgba = verdict.get("rgba") or ctx.get_extra("rgba")

            final_result = {
                "status": verdict.get("status", "ok"),
                "analysis_mode": analysis_mode,
                "judge_mode": verdict.get("judge_mode", "standard"),
                "rgba": rgba,
                "sources": sources,
                "rationale": verdict.get("rationale"),
                "analysis": verdict.get("analysis") or verdict.get("rationale"),
                "verified_score": verdict.get("verified_score"),
                "explainability_score": verdict.get("explainability_score"),
                "danger_score": verdict.get("danger_score"),
                "context_score": verdict.get("context_score"),
                "style_score": verdict.get("style_score"),
                "confidence_score": verdict.get("confidence_score"),
                "bias_score": verdict.get("style_score"),
                "cost": verdict.get("cost"),
            }

            # Text: prefer prepared_text (extracted/cleaned), fallback to raw input.
            final_result["text"] = prepared_text or original_fact or ctx.get_extra("input_text") or ""
            final_result["fact"] = original_fact
            final_result["original_fact"] = original_fact

            claim_verdicts = verdict.get("claim_verdicts") or []
            claim_text_by_id = _build_claim_text_lookup(claims_contract)

            # Enrich verdict entries with claim text when missing.
            enriched_verdicts = []
            for cv in claim_verdicts:
                if not isinstance(cv, dict):
                    continue
                cid = str(cv.get("claim_id") or cv.get("id") or "")
                if cid and not (cv.get("text") or cv.get("claim_text")):
                    fallback_txt = claim_text_by_id.get(cid)
                    if fallback_txt:
                        cv = dict(cv)
                        cv["text"] = fallback_txt
                enriched_verdicts.append(cv)

            final_result["claim_verdicts"] = enriched_verdicts

            details = []
            for cv in enriched_verdicts:
                if not isinstance(cv, dict):
                    continue
                text = (cv.get("text") or cv.get("claim_text") or cv.get("claim") or "").strip()
                if not text:
                    continue
                details.append(
                    {
                        "text": text,
                        "rgba": cv.get("rgba") or rgba,
                        "rationale": cv.get("rationale") or verdict.get("rationale"),
                        "sources": cv.get("sources") or sources,
                    }
                )

            if not details and claims_contract:
                details = [
                    {"text": claim.text, "rgba": rgba, "rationale": verdict.get("rationale"), "sources": sources}
                    for claim in claims_contract.claims
                    if claim.text
                ]

            final_result["details"] = details

            final_result["_extracted_claims"] = ctx.claims or []

            anchor = _build_anchor_claim(
                verdict, claims_contract, claim_text_by_id, ctx.claims or []
            )
            if anchor:
                final_result["anchor_claim"] = anchor

            Trace.event("result_assembly.completed", {"judge_mode": "standard"})

            return ctx.set_extra("final_result", final_result)

        except Exception as e:
            logger.exception("[AssembleStandardResultStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


@dataclass
class ResultAssemblyStep(AssembleStandardResultStep):
    """Backward-compatible alias for standard result assembly."""

    name: str = "result_assembly"
