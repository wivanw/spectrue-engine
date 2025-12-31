# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

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
from spectrue_core.verification.claim_frame_builder import (
    build_claim_frames_from_pipeline,
)
from spectrue_core.verification.execution_plan import ClaimExecutionState


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
    
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("build_claim_frames")
        
        try:
            # Get required data from context
            claims = ctx.extras.get("claims", [])
            document_text = ctx.extras.get("clean_text", "") or ctx.extras.get("input_text", "")
            evidence_by_claim = ctx.extras.get("evidence_by_claim", {})
            execution_states = ctx.extras.get("execution_states", {})
            
            if not claims:
                Trace.event("build_claim_frames.skip", {"reason": "no_claims"})
                return ctx.with_extras(deep_claim_ctx=DeepClaimContext())
            
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
            
            return ctx.with_extras(deep_claim_ctx=deep_ctx)
            
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
    
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
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
            
            return ctx.with_extras(deep_claim_ctx=deep_ctx)
            
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
    
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
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
            
            return ctx.with_extras(deep_claim_ctx=deep_ctx)
            
        finally:
            Trace.phase_end("judge_claims")


class AssembleDeepResultStep(Step):
    """
    Step that assembles final deep analysis result.
    
    Combines ClaimFrames with JudgeOutputs to produce ClaimResults.
    Returns DeepAnalysisResult with no aggregate verdict.
    """
    
    @property
    def name(self) -> str:
        return "assemble_deep_result"
    
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
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
            
            # Build final result
            deep_result = DeepAnalysisResult(
                analysis_mode="deep",
                claim_results=tuple(claim_results),
            )
            
            Trace.event("assemble_deep_result.complete", {
                "result_count": len(claim_results),
            })
            
            return ctx.with_extras(
                deep_claim_ctx=deep_ctx,
                deep_analysis_result=deep_result,
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
