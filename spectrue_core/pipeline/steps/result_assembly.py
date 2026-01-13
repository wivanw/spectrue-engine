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
import math
from decimal import Decimal
from dataclasses import dataclass

from spectrue_core.pipeline.contracts import (
    CLAIMS_KEY,
    INPUT_DOC_KEY,
    RGBA_AUDIT_KEY,
    Claims,
    InputDoc,
    RGBAAuditResultPayload,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.schema.rgba_audit import RGBAResult
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
    if isinstance(anchor, dict):
        anchor_id = (
            anchor.get("id")
            or anchor.get("claim_id")
            or (claims_contract.anchor_claim_id if claims_contract else None)
        )
        if not anchor_id and fallback_claims:
            first = fallback_claims[0]
            if isinstance(first, dict):
                anchor_id = first.get("id") or first.get("claim_id")

        anchor_text = anchor.get("text") or anchor.get("normalized_text")
        if not anchor_text and anchor_id:
            anchor_text = claim_text_by_id.get(str(anchor_id))

        if anchor_id and not anchor_text and fallback_claims:
            first = fallback_claims[0]
            if isinstance(first, dict):
                anchor_text = (first.get("normalized_text") or first.get("text") or "").strip()

        if anchor_id and anchor_text:
            normalized = dict(anchor)
            normalized["id"] = str(anchor_id)
            normalized["text"] = anchor_text
            return normalized

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


def _extract_credits(cost_summary: dict | None) -> int | None:
    if not isinstance(cost_summary, dict):
        return None
    credits = cost_summary.get("credits_used")
    if credits is None:
        credits = cost_summary.get("total_credits")
    if isinstance(credits, (int, float, Decimal)):
        return int(math.ceil(float(credits)))
    return None


def _coerce_score(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


def _build_rgba(verdict: dict, ctx: PipelineContext) -> list[float]:
    rgba = verdict.get("rgba") or ctx.get_extra("rgba")
    if (
        isinstance(rgba, list)
        and len(rgba) == 4
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rgba)
    ):
        return [float(x) for x in rgba]

    danger = _coerce_score(verdict.get("danger_score"), -1.0)
    verified = _coerce_score(verdict.get("verified_score"), -1.0)
    style = verdict.get("style_score")
    if not isinstance(style, (int, float)) or isinstance(style, bool):
        style = verdict.get("context_score")
    honesty = _coerce_score(style, -1.0)
    explainability = verdict.get("explainability_score")
    if not isinstance(explainability, (int, float)) or isinstance(explainability, bool):
        explainability = verdict.get("confidence_score")
    explainability_score = _coerce_score(explainability, -1.0)

    return [danger, verified, honesty, explainability_score]


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

            rgba = _build_rgba(verdict, ctx)
            rgba_audit_payload = None
            rgba_audit = ctx.get_extra(RGBA_AUDIT_KEY)
            if isinstance(rgba_audit, RGBAResult):
                rgba_audit_payload = RGBAAuditResultPayload.from_result(rgba_audit).to_payload()
            elif isinstance(rgba_audit, RGBAAuditResultPayload):
                rgba_audit_payload = rgba_audit.to_payload()
            elif isinstance(rgba_audit, dict):
                rgba_audit_payload = rgba_audit
            cost_summary = ctx.get_extra("cost_summary")
            if not isinstance(cost_summary, dict):
                cost_summary = None

            final_result = {
                "status": verdict.get("status", "ok"),
                "analysis_mode": analysis_mode,
                "judge_mode": verdict.get("judge_mode", "standard"),
                "rgba": rgba,
                "sources": sources,
                "rationale": verdict.get("rationale"),
                "analysis": verdict.get("analysis") or verdict.get("rationale"),
                "analysis": verdict.get("analysis") or verdict.get("rationale"),
                "verified_score": (_coerce_score(verdict.get("verified_score"), 0.0) + 1.0) / 2.0,  # Normalize [-1, 1] -> [0, 1]
                "veracity_signed": verdict.get("verified_score"),  # Preserve signed score for debug/backend use
                "explainability_score": verdict.get("explainability_score"),
                "danger_score": verdict.get("danger_score"),
                "context_score": verdict.get("context_score"),
                "style_score": verdict.get("style_score"),
                "confidence_score": verdict.get("confidence_score"),
                "bias_score": verdict.get("style_score"),
                "cost": verdict.get("cost"),
            }
            if rgba_audit_payload is not None:
                final_result["rgba_audit"] = rgba_audit_payload
            if cost_summary is not None:
                final_result["cost_summary"] = cost_summary
                credits = _extract_credits(cost_summary)
                if credits is not None:
                    final_result["credits"] = credits

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

            # Standard mode contract:
            # 1. No "accordion" of claims -> claim_verdicts = []
            # 2. UI card renders via 'details' -> details = [single_global_entry]
            
            final_result["claim_verdicts"] = []  # Hide specific claim list in UI
            
            # Construct single detail entry for validity visualization
            anchor_text = ""
            anchor_obj = _build_anchor_claim(
                verdict, claims_contract, claim_text_by_id, ctx.claims or []
            )
            if anchor_obj:
                anchor_text = anchor_obj.get("text", "")
            
            display_text = anchor_text or final_result["text"] or ""
            
            final_result["details"] = [
                {
                    "text": display_text,
                    "rgba": rgba,
                    "sources": sources,
                    "rationale": final_result.get("rationale"),
                    "verdict": final_result.get("status"),
                }
            ]

            final_result["_extracted_claims"] = ctx.claims or []

            anchor = _build_anchor_claim(
                verdict, claims_contract, claim_text_by_id, ctx.claims or []
            )
            if anchor:
                final_result["anchor_claim"] = anchor

            Trace.event(
                "final_result.keys",
                {
                    "judge_mode": final_result.get("judge_mode", "standard"),
                    "keys": sorted(final_result.keys()),
                },
            )

            Trace.event("result_assembly.completed", {"judge_mode": "standard"})

            return ctx.set_extra("final_result", final_result)

        except Exception as e:
            logger.exception("[AssembleStandardResultStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e


@dataclass
class ResultAssemblyStep(AssembleStandardResultStep):
    """Backward-compatible alias for standard result assembly."""

    name: str = "result_assembly"
