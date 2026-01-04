# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Deep claim pipeline steps for per-claim judging.

These steps implement the deep analysis mode where each claim is
evaluated independently with its own ClaimFrame and JudgeOutput.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.skills.claim_judge import ClaimJudgeSkill
from spectrue_core.agents.skills.evidence_summarizer import EvidenceSummarizerSkill
from spectrue_core.pipeline.contracts import JUDGMENTS_KEY, Judgments
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    EvidenceSummary,
    JudgeOutput,
)
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.claim_frame_builder import (
    build_claim_frames_from_pipeline,
)


@dataclass
class DeepClaimContext:
    """
    Context for deep claim processing.
    
    Tracks claim frames, summaries, and results through the pipeline.
    """
    claim_frames: list[ClaimFrame] = field(default_factory=list)
    evidence_summaries: dict[str, EvidenceSummary] = field(default_factory=dict)
    judge_outputs: dict[str, JudgeOutput] = field(default_factory=dict)
    claim_results: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)


class BuildClaimFramesStep(Step):
    """
    Step that builds ClaimFrame objects for each extracted claim.
    
    Converts pipeline state (claims, evidence, execution state) into
    per-claim ClaimFrame bundles.
    """

    @property
    def name(self) -> str:
        return "build_claim_frames"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("build_claim_frames")

        try:
            # Get required data from context
            # Claims are stored in ctx.claims by ExtractClaimsStep
            claims = ctx.claims or []
            document_text = ctx.extras.get("clean_text", "") or ctx.extras.get("input_text", "") or ctx.extras.get("prepared_fact", "")
            evidence_by_claim = ctx.extras.get("evidence_by_claim", {})
            execution_states = ctx.extras.get("execution_states", {})

            if not claims:
                Trace.event("build_claim_frames.skip", {"reason": "no_claims"})
                return ctx.set_extra("deep_claim_ctx", DeepClaimContext())

            # Build frames
            frames = build_claim_frames_from_pipeline(
                claims=claims,
                document_text=document_text,
                evidence_by_claim=evidence_by_claim,
                execution_states=execution_states,
            )

            deep_ctx = DeepClaimContext(claim_frames=frames)

            Trace.event("build_claim_frames.complete", {
                "frame_count": len(frames),
                "claim_ids": [f.claim_id for f in frames],
            })

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("build_claim_frames")


class SummarizeEvidenceStep(Step):
    """
    Step that summarizes evidence for each claim.
    
    Uses EvidenceSummarizerSkill to categorize evidence by stance.
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    @property
    def name(self) -> str:
        return "summarize_evidence"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("summarize_evidence")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())

            if not deep_ctx.claim_frames:
                Trace.event("summarize_evidence.skip", {"reason": "no_frames"})
                return ctx

            skill = EvidenceSummarizerSkill(self._llm)

            # Process all claims in parallel
            async def summarize_one(frame: ClaimFrame) -> tuple[str, EvidenceSummary]:
                summary = await skill.summarize(frame)
                return frame.claim_id, summary

            tasks = [summarize_one(frame) for frame in deep_ctx.claim_frames]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            summaries: dict[str, EvidenceSummary] = {}
            for result in results:
                if isinstance(result, Exception):
                    Trace.event("summarize_evidence.task_error", {"error": str(result)})
                    continue
                claim_id, summary = result
                summaries[claim_id] = summary

            deep_ctx.evidence_summaries = summaries

            Trace.event("summarize_evidence.complete", {
                "summary_count": len(summaries),
            })

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("summarize_evidence")


class JudgeClaimsStep(Step):
    """
    Step that produces verdicts for each claim.
    
    Uses ClaimJudgeSkill to generate RGBA scores and verdicts.
    Output is returned unchanged to the frontend.
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    @property
    def name(self) -> str:
        return "judge_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("judge_claims")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())

            if not deep_ctx.claim_frames:
                Trace.event("judge_claims.skip", {"reason": "no_frames"})
                return ctx

            skill = ClaimJudgeSkill(self._llm)
            
            # Get UI locale from pipeline context
            # This is the user's interface language from the API request
            ui_locale = ctx.lang or "en"

            # Process all claims in parallel
            async def judge_one(frame: ClaimFrame) -> tuple[str, JudgeOutput | None, dict[str, Any] | None]:
                summary = deep_ctx.evidence_summaries.get(frame.claim_id)
                try:
                    Trace.event(
                        "judge_claims.invoked",
                        {
                            "claim_id": frame.claim_id,
                            "has_summary": summary is not None,
                            "ui_locale": ui_locale,
                        },
                    )
                    # Pass ui_locale to generate explanation in user's language
                    output = await skill.judge(frame, summary, ui_locale=ui_locale)
                    return frame.claim_id, output, None
                except Exception as e:
                    return frame.claim_id, None, {
                        "error_type": "llm_failed",
                        "message": str(e),
                    }

            tasks = [judge_one(frame) for frame in deep_ctx.claim_frames]
            results = await asyncio.gather(*tasks)

            outputs: dict[str, JudgeOutput] = {}
            errors: dict[str, dict[str, Any]] = {}
            for claim_id, output, error in results:
                if error:
                    Trace.event("judge_claims.task_error", {"claim_id": claim_id, "error": error.get("message")})
                    errors[claim_id] = error
                    continue
                if output:
                    outputs[claim_id] = output

            deep_ctx.judge_outputs = outputs
            deep_ctx.errors = errors

            Trace.event("judge_claims.complete", {
                "output_count": len(outputs),
                "verdicts": {cid: out.verdict for cid, out in outputs.items()},
                "ui_locale": ui_locale,
            })

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("judge_claims")


class AssembleDeepResultStep(Step):
    """
    Step that assembles final deep analysis result.

    Produces per-claim outputs with no global scoring.
    """

    @property
    def name(self) -> str:
        return "assemble_deep_result"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("assemble_deep_result")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())

            analysis_mode = "deep"
            judge_mode = "deep"

            claim_results: list[dict[str, Any]] = []
            details_for_frontend: list[dict[str, Any]] = []
            claim_verdicts: list[dict[str, Any]] = []

            from spectrue_core.utils.trust_utils import enrich_sources_with_trust

            for frame in deep_ctx.claim_frames:
                judge_output = deep_ctx.judge_outputs.get(frame.claim_id)
                error = deep_ctx.errors.get(frame.claim_id)

                if error or judge_output is None:
                    error_payload = error or {"error_type": "judge_missing", "message": "Judge output missing"}
                    claim_results.append({
                        "claim_id": frame.claim_id,
                        "status": "error",
                        "rgba": None,
                        "explanation": None,
                        "sources_used": [],
                        "error": error_payload,
                    })
                    details_for_frontend.append({
                        "text": frame.claim_text,
                        "error_key": error_payload.get("error_type", "judge_failed"),
                    })
                    continue

                rgba = [
                    judge_output.rgba.r,
                    judge_output.rgba.g,
                    judge_output.rgba.b,
                    judge_output.rgba.a,
                ]
                sources_used = list(judge_output.sources_used or [])

                claim_results.append({
                    "claim_id": frame.claim_id,
                    "status": "ok",
                    "rgba": rgba,
                    "explanation": judge_output.explanation,
                    "sources_used": sources_used,
                })

                sources_list = []
                evidence_map = {ei.evidence_id: ei for ei in frame.evidence_items}
                url_map = {ei.url: ei for ei in frame.evidence_items if ei.url}

                for src_ref in sources_used:
                    ei = evidence_map.get(src_ref) or url_map.get(src_ref)
                    if ei:
                        sources_list.append({
                            "url": ei.url,
                            "domain": ei.source_type or "web",
                            "title": ei.title,
                            "citation_text": ei.snippet,
                        })

                if not sources_list and frame.evidence_items:
                    for ei in frame.evidence_items:
                        sources_list.append({
                            "url": ei.url,
                            "domain": ei.source_type or "web",
                            "title": ei.title,
                            "citation_text": ei.snippet,
                        })

                sources_list = enrich_sources_with_trust(sources_list)

                details_for_frontend.append({
                    "text": frame.claim_text,
                    "rgba": rgba,
                    "rationale": judge_output.explanation or f"Verdict: {judge_output.verdict}",
                    "sources": sources_list,
                    "verified_score": judge_output.rgba.g,
                    "danger_score": judge_output.rgba.r,
                })

                claim_verdicts.append({
                    "claim_id": frame.claim_id,
                    "text": frame.claim_text,
                    "rgba": rgba,
                    "verdict": judge_output.verdict,
                    "confidence": judge_output.confidence,
                    "explanation": judge_output.explanation,
                    "sources": sources_list,
                })

            deep_ctx.claim_results = claim_results

            verdict = {
                "judge_mode": judge_mode,
                "claim_verdicts": claim_verdicts,
            }

            final_result = {
                "analysis_mode": analysis_mode,
                "judge_mode": judge_mode,
                "deep_analysis": {
                    "claim_results": claim_results,
                },
                "details": details_for_frontend,
                "claim_verdicts": claim_verdicts,
            }

            Trace.event("assemble_deep_result.complete", {"result_count": len(claim_results)})

            judgments = Judgments(standard=None, deep=tuple(claim_results))

            return (
                ctx.with_update(verdict=verdict)
                .set_extra("deep_claim_ctx", deep_ctx)
                .set_extra("deep_analysis_result", final_result.get("deep_analysis"))
                .set_extra("final_result", final_result)
                .set_extra(JUDGMENTS_KEY, judgments)
            )

        finally:
            Trace.phase_end("assemble_deep_result")


def get_deep_claim_steps(llm_client: LLMClient) -> list[Step]:
    """
    Get all steps for deep claim processing.
    
    Args:
        llm_client: LLM client for skill calls
    
    Returns:
        List of steps in execution order
    """
    return [
        BuildClaimFramesStep(),
        SummarizeEvidenceStep(llm_client),
        JudgeClaimsStep(llm_client),
        AssembleDeepResultStep(),
    ]
