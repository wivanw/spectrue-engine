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
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    ClaimResult,
    DeepAnalysisResult,
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
    claim_results: list[ClaimResult] = field(default_factory=list)
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

            # Process all claims in parallel
            async def judge_one(frame: ClaimFrame) -> tuple[str, JudgeOutput]:
                summary = deep_ctx.evidence_summaries.get(frame.claim_id)
                output = await skill.judge(frame, summary)
                return frame.claim_id, output

            tasks = [judge_one(frame) for frame in deep_ctx.claim_frames]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            outputs: dict[str, JudgeOutput] = {}
            for result in results:
                if isinstance(result, Exception):
                    Trace.event("judge_claims.task_error", {"error": str(result)})
                    continue
                claim_id, output = result
                outputs[claim_id] = output

            deep_ctx.judge_outputs = outputs

            Trace.event("judge_claims.complete", {
                "output_count": len(outputs),
                "verdicts": {cid: out.verdict for cid, out in outputs.items()},
            })

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("judge_claims")


class AssembleDeepResultStep(Step):
    """
    Step that assembles final deep analysis result.
    
    Combines ClaimFrames with JudgeOutputs to produce ClaimResults.
    Produces final_result with deep_analysis field directly included.
    No aggregate verdict is computed - that's for standard mode only.
    """

    @property
    def name(self) -> str:
        return "assemble_deep_result"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("assemble_deep_result")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())

            claim_results: list[ClaimResult] = []

            for frame in deep_ctx.claim_frames:
                judge_output = deep_ctx.judge_outputs.get(frame.claim_id)
                evidence_summary = deep_ctx.evidence_summaries.get(frame.claim_id)
                error = deep_ctx.errors.get(frame.claim_id)

                if judge_output is None:
                    # Create error result if no judge output
                    from spectrue_core.schema.claim_frame import RGBAScore
                    judge_output = JudgeOutput(
                        claim_id=frame.claim_id,
                        rgba=RGBAScore(r=0.0, g=0.5, b=0.5, a=0.1),
                        confidence=0.1,
                        verdict="NEI",
                        explanation="No judge output available",
                        sources_used=(),
                        missing_evidence=(),
                    )
                    error = error or {"message": "Judge output missing"}

                claim_result = ClaimResult(
                    claim_frame=frame,
                    judge_output=judge_output,
                    evidence_summary=evidence_summary,
                    error=error,
                )
                claim_results.append(claim_result)

            deep_ctx.claim_results = claim_results

            # Build deep_analysis structure for direct inclusion in final_result
            deep_analysis = DeepAnalysisResult(
                analysis_mode="deep",
                claim_results=tuple(claim_results),
            )

            # Serialize claim_results for API response
            claim_results_serialized = [
                self._serialize_claim_result(cr) for cr in claim_results
            ]

            # Get existing final_result or create new one
            existing_final = ctx.extras.get("final_result", {})

            # CRITICAL: Update claim_verdicts in verdict with RGBA from judge_outputs
            # This ensures pipeline_evidence doesn't overwrite per-claim RGBA with globals
            verdict = ctx.verdict or {}
            claim_verdicts = verdict.get("claim_verdicts")
            
            # If claim_verdicts is empty (deep mode default behavior since EvidenceFlow
            # returns empty list in collect_only mode), we must build it from scratch
            # using the ClaimResults we just assembled.
            if not claim_verdicts:
                claim_verdicts = []
                for cr in claim_results:
                    frame = cr.claim_frame
                    judge = cr.judge_output
                    
                    # Map sources used by judge back to full source objects
                    # We look up in frame.evidence_items
                    used_sources = []
                    evidence_map = {ei.evidence_id: ei for ei in frame.evidence_items}
                    
                    for src_ref in (judge.sources_used or []):
                        # src_ref likely evidence_id or URL
                        if src_ref in evidence_map:
                            ei = evidence_map[src_ref]
                            used_sources.append({
                                "url": ei.url,
                                "domain": ei.source_type or "web", # Fallback
                                "title": ei.title,
                                "citation_text": ei.snippet,
                                "trust_category": ei.source_tier,
                                "is_trusted": ei.source_tier in ("tier1", "tier2"),
                            })

                    cv = {
                        "claim_id": frame.claim_id,
                        "text": frame.claim_text,
                        "rgba": [
                            judge.rgba.r,
                            judge.rgba.g,
                            judge.rgba.b,
                            judge.rgba.a,
                        ],
                        "verdict": judge.verdict,
                        "confidence": judge.confidence,
                        "explanation": judge.explanation,
                        "sources": used_sources,
                    }
                    claim_verdicts.append(cv)
                verdict["claim_verdicts"] = claim_verdicts

            elif isinstance(claim_verdicts, list):
                # Update existing claim_verdicts (if any existed, e.g. from legacy flow)
                for cv in claim_verdicts:
                    if not isinstance(cv, dict):
                        continue
                    cid = cv.get("claim_id")
                    if cid and cid in deep_ctx.judge_outputs:
                        judge_out = deep_ctx.judge_outputs[cid]
                        # Copy RGBA from judge_output to claim_verdict
                        # This is the canonical RGBA for deep mode
                        cv["rgba"] = [
                            judge_out.rgba.r,
                            judge_out.rgba.g,
                            judge_out.rgba.b,
                            judge_out.rgba.a,
                        ]
                        cv["verdict"] = judge_out.verdict
                        cv["confidence"] = judge_out.confidence
                        cv["explanation"] = judge_out.explanation
                        Trace.event(
                            "deep.claim_verdict_rgba_updated",
                            {
                                "claim_id": cid,
                                "rgba": cv["rgba"],
                                "verdict": cv["verdict"],
                            },
                        )
                verdict["claim_verdicts"] = claim_verdicts

            # Add deep_analysis to final_result - no conditional logic needed
            # The API layer will just pass this through
            # 
            # IMPORTANT: Deep mode does NOT populate global scores (danger_score, style_score, etc.)
            # These are computed per-claim and stored in claim_results[].judge_output.rgba
            # The UI MUST read from deep_analysis.claim_results, not from global fields
            
            # Import trust enrichment
            from spectrue_core.utils.trust_utils import enrich_sources_with_trust
            
            # Build details in format expected by frontend
            # Frontend expects: {text, rgba, rationale, sources, verified_score, danger_score}
            details_for_frontend = []
            for cr in claim_results:
                frame = cr.claim_frame
                judge = cr.judge_output
                
                # Build sources list
                sources_list = []
                evidence_map = {ei.evidence_id: ei for ei in frame.evidence_items}
                # Also map by URL for fallback matching
                url_map = {ei.url: ei for ei in frame.evidence_items if ei.url}
                
                for src_ref in (judge.sources_used or []):
                    # Try to find by evidence_id first, then by URL
                    ei = evidence_map.get(src_ref) or url_map.get(src_ref)
                    if ei:
                        sources_list.append({
                            "url": ei.url,
                            "domain": ei.source_type or "web",
                            "title": ei.title,
                            "citation_text": ei.snippet,
                        })
                
                # If no sources matched, include all evidence items as sources
                # This ensures frontend always has sources to display
                if not sources_list and frame.evidence_items:
                    for ei in frame.evidence_items:
                        sources_list.append({
                            "url": ei.url,
                            "domain": ei.source_type or "web",
                            "title": ei.title,
                            "citation_text": ei.snippet,
                        })
                
                # Enrich sources with trust_category from TRUSTED_SOURCES registry
                # This maps domain -> category (e.g., bbc.com -> general_news_western)
                sources_list = enrich_sources_with_trust(sources_list)
                
                details_for_frontend.append({
                    "text": frame.claim_text,
                    "rgba": [
                        judge.rgba.r,
                        judge.rgba.g,
                        judge.rgba.b,
                        judge.rgba.a,
                    ],
                    "rationale": judge.explanation or f"Verdict: {judge.verdict}",
                    "sources": sources_list,
                    "verified_score": judge.rgba.g,  # G component = verified score
                    "danger_score": judge.rgba.r,    # R component = danger score
                })
            
            final_result = {
                **existing_final,
                "analysis_mode": "deep",
                "judge_mode": "deep",  # Explicit marker for UI
                "deep_analysis": {
                    "analysis_mode": "deep",
                    "claim_results": claim_results_serialized,
                },
                # For UI compatibility, populate 'details' with the claim verdicts
                # This ensures the main sentence list in the UI renders correctly
                "details": details_for_frontend,
                "claim_verdicts": verdict.get("claim_verdicts", []),  # Keep for API compatibility
                # NOTE: We intentionally do NOT set these global fields in deep mode:
                # - danger_score
                # - style_score  
                # - explainability_score
                # - verified_score (as global)
                # These are ONLY valid in standard mode.
                # Deep mode has per-claim RGBA in deep_analysis.claim_results[].judge_output.rgba
                "cost": (ctx.verdict or {}).get("cost", 0.0),
            }

            Trace.event("assemble_deep_result.complete", {
                "result_count": len(claim_results),
            })

            # Return updated context with:
            # - verdict containing claim_verdicts with per-claim RGBA from judge
            # - final_result containing deep_analysis
            return ctx.with_update(verdict=verdict).set_extra(
                "deep_claim_ctx", deep_ctx
            ).set_extra(
                "deep_analysis_result", deep_analysis
            ).set_extra(
                "final_result", final_result
            )

        finally:
            Trace.phase_end("assemble_deep_result")

    def _serialize_claim_result(self, cr: ClaimResult) -> dict:
        """Serialize ClaimResult to dict for API response."""
        frame = cr.claim_frame
        judge = cr.judge_output
        summary = cr.evidence_summary

        return {
            "claim_frame": {
                "claim_id": frame.claim_id,
                "claim_text": frame.claim_text,
                "claim_language": frame.claim_language,
                "context_excerpt": {
                    "text": frame.context_excerpt.text,
                    "source_type": frame.context_excerpt.source_type,
                    "span_start": frame.context_excerpt.span_start,
                    "span_end": frame.context_excerpt.span_end,
                },
                "context_meta": {
                    "document_id": frame.context_meta.document_id,
                    "paragraph_index": frame.context_meta.paragraph_index,
                    "sentence_index": frame.context_meta.sentence_index,
                    "sentence_window": frame.context_meta.sentence_window,
                },
                "evidence_items": [
                    {
                        "evidence_id": ei.evidence_id,
                        "claim_id": ei.claim_id,
                        "url": ei.url,
                        "title": ei.title,
                        "source_tier": ei.source_tier,
                        "source_type": ei.source_type,
                        "stance": ei.stance,
                        "quote": ei.quote,
                        "snippet": ei.snippet,
                        "relevance": ei.relevance,
                    }
                    for ei in frame.evidence_items
                ],
                "evidence_stats": {
                    "total_sources": frame.evidence_stats.total_sources,
                    "support_sources": frame.evidence_stats.support_sources,
                    "refute_sources": frame.evidence_stats.refute_sources,
                    "context_sources": frame.evidence_stats.context_sources,
                    "high_trust_sources": frame.evidence_stats.high_trust_sources,
                    "direct_quotes": frame.evidence_stats.direct_quotes,
                    "conflicting_evidence": frame.evidence_stats.conflicting_evidence,
                    "missing_sources": frame.evidence_stats.missing_sources,
                    "missing_direct_quotes": frame.evidence_stats.missing_direct_quotes,
                },
                "retrieval_trace": {
                    "phases_completed": list(frame.retrieval_trace.phases_completed),
                    "hops": [
                        {
                            "hop_index": h.hop_index,
                            "query": h.query,
                            "decision": h.decision,
                            "reason": h.reason,
                            "phase_id": h.phase_id,
                            "query_type": h.query_type,
                            "results_count": h.results_count,
                        }
                        for h in frame.retrieval_trace.hops
                    ],
                    "stop_reason": frame.retrieval_trace.stop_reason,
                    "sufficiency_reason": frame.retrieval_trace.sufficiency_reason,
                },
            },
            "judge_output": {
                "claim_id": judge.claim_id,
                "rgba": judge.rgba.to_dict(),
                "confidence": judge.confidence,
                "verdict": judge.verdict,
                "explanation": judge.explanation,
                "sources_used": list(judge.sources_used),
                "missing_evidence": list(judge.missing_evidence),
            },
            "evidence_summary": {
                "supporting_evidence": [
                    {"evidence_id": ref.evidence_id, "reason": ref.reason}
                    for ref in (summary.supporting_evidence if summary else ())
                ],
                "refuting_evidence": [
                    {"evidence_id": ref.evidence_id, "reason": ref.reason}
                    for ref in (summary.refuting_evidence if summary else ())
                ],
                "contextual_evidence": [
                    {"evidence_id": ref.evidence_id, "reason": ref.reason}
                    for ref in (summary.contextual_evidence if summary else ())
                ],
                "evidence_gaps": list(summary.evidence_gaps if summary else ()),
                "conflicts_present": summary.conflicts_present if summary else False,
            } if summary else None,
            "error": cr.error,
        }



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