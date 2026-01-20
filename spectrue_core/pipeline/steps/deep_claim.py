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
import re
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.agents.llm_client import LLMClient, is_schema_failure
from spectrue_core.agents.llm_schemas import CLAIM_JUDGE_SCHEMA
from spectrue_core.agents.skills.claim_judge import ClaimJudgeSkill
from spectrue_core.agents.skills.claim_judge_prompts import (
    build_claim_judge_prompt,
    build_claim_judge_system_prompt,
)
from spectrue_core.verification.scoring.judge_evidence_stats import build_judge_evidence_stats
from spectrue_core.agents.skills.evidence_summarizer import EvidenceSummarizerSkill
from spectrue_core.pipeline.mode import ScoringMode
from spectrue_core.pipeline.contracts import (
    JUDGMENTS_KEY,
    RGBA_AUDIT_KEY,
    Judgments,
    RGBAAuditResultPayload,
)
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    EvidenceSummary,
    JudgeOutput,
)
from spectrue_core.schema.rgba_audit import RGBAResult
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.claim_frame_builder import (
    build_claim_frames_from_pipeline,
)
from spectrue_core.llm.model_registry import ModelID


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


_SCHEMA_MISSING_RE = re.compile(r"\$\.(?P<field>[A-Za-z0-9_\\[\\].]+): missing required field")


def _root_cause(exc: Exception) -> Exception:
    seen: set[int] = set()
    current: Exception = exc
    while True:
        next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if not isinstance(next_exc, Exception):
            return current
        next_id = id(next_exc)
        if next_id in seen:
            return current
        seen.add(next_id)
        current = next_exc


def _is_format_error(exc: Exception) -> bool:
    if is_schema_failure(exc):
        return True
    msg = str(exc).lower()
    return "json parse" in msg or "invalid json" in msg


def _extract_missing_fields(message: str) -> list[str]:
    if not message:
        return []
    return list({match.group("field") for match in _SCHEMA_MISSING_RE.finditer(message)})


def _build_error_payload(
    *,
    error_type: str,
    message: str,
    missing_fields: list[str] | None = None,
    repair_attempted: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"error_type": error_type, "message": message}
    if missing_fields:
        payload["missing_fields"] = missing_fields
    if repair_attempted:
        payload["repair_attempted"] = True
    return payload


class BuildClaimFramesStep(Step):
    """
    Step that builds ClaimFrame objects for each extracted claim.
    
    Converts pipeline state (claims, evidence, execution state) into
    per-claim ClaimFrame bundles.
    """
    weight: float = 1.0

    name: str = "build_claim_frames"

    def __init__(self, config: Any | None = None):
        self._config = config

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("build_claim_frames")

        try:
            # Get required data from context
            # Claims are stored in ctx.claims by ExtractClaimsStep
            claims = ctx.claims or []
            document_text = ctx.extras.get("clean_text", "") or ctx.extras.get("input_text", "") or ctx.extras.get("prepared_fact", "")
            evidence_by_claim = ctx.extras.get("evidence_by_claim", {})
            execution_states = ctx.extras.get("execution_states", {})
            corroboration_by_claim = ctx.get_extra("corroboration_by_claim")

            if not claims:
                Trace.event("build_claim_frames.skip", {"reason": "no_claims"})
                return ctx.set_extra("deep_claim_ctx", DeepClaimContext())

            confirmation_lambda = None
            if ctx.mode.api_analysis_mode == AnalysisMode.DEEP_V2:
                from spectrue_core.runtime_config import DeepV2Config
                runtime = getattr(self._config, "runtime", None)
                deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())
                confirmation_lambda = deep_v2_cfg.confirmation_lambda

            # Build frames
            frames = build_claim_frames_from_pipeline(
                claims=claims,
                document_text=document_text,
                evidence_by_claim=evidence_by_claim,
                execution_states=execution_states,
                confirmation_lambda=confirmation_lambda,
                corroboration_by_claim=corroboration_by_claim if isinstance(corroboration_by_claim, dict) else None,
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
    weight: float = 5.0

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    name: str = "summarize_evidence"

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
    weight: float = 20.0

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    name: str = "judge_claims"

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
            analysis_mode = ctx.mode.api_analysis_mode

            corr_by_claim = ctx.get_extra("corroboration_by_claim") or {}
            est_by_claim = ctx.get_extra("evidence_stats_by_claim") or {}

            async def _repair_claim_output(
                frame: ClaimFrame,
                summary: EvidenceSummary | None,
            ) -> JudgeOutput:
                base_prompt = build_claim_judge_prompt(
                    frame,
                    summary,
                    ui_locale=ui_locale,
                    analysis_mode=analysis_mode,
                )
                repair_prompt = (
                    "Your previous response was invalid or missing required fields. "
                    "Return ONLY valid JSON matching the schema with keys: "
                    "claim_id, rgba{R,G,B,A}, confidence, verdict, explanation, "
                    "sources_used, missing_evidence.\n\n"
                    f"{base_prompt}"
                )
                repair_system = build_claim_judge_system_prompt(lang=ui_locale)
                repair_system = f"{repair_system}\nReturn only JSON; no markdown or extra text."

                response = await self._llm.call_json(
                    model=self._llm.model or ModelID.NANO,
                    input=repair_prompt,
                    instructions=repair_system,
                    response_schema=CLAIM_JUDGE_SCHEMA,
                    reasoning_effort="low",
                    trace_kind="claim_judge.repair",
                )

                repaired = skill._parse_response(response, frame)
                return skill._validate_sources_used(repaired, frame)

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
                    evidence_stats = build_judge_evidence_stats(
                        claim_id=frame.claim_id,
                        corroboration_by_claim=corr_by_claim if isinstance(corr_by_claim, dict) else None,
                        evidence_stats_by_claim=est_by_claim if isinstance(est_by_claim, dict) else None,
                    )

                    output = await skill.judge(
                        frame,
                        summary,
                        ui_locale=ui_locale,
                        analysis_mode=analysis_mode,
                        evidence_stats=evidence_stats,
                    )
                    return frame.claim_id, output, None
                except Exception as e:
                    root = _root_cause(e)
                    message = str(root)
                    missing_fields = _extract_missing_fields(message)

                    if _is_format_error(root):
                        Trace.event(
                            "judge_claims.schema_mismatch",
                            {
                                "claim_id": frame.claim_id,
                                "missing_fields": missing_fields,
                                "error": message[:300],
                            },
                        )
                        try:
                            Trace.event(
                                "judge_claims.repair_needed",
                                {"claim_id": frame.claim_id, "missing_fields": missing_fields},
                            )
                            repaired = await _repair_claim_output(frame, summary)
                            Trace.event(
                                "judge_claims.repair_succeeded",
                                {"claim_id": frame.claim_id},
                            )
                            return frame.claim_id, repaired, None
                        except Exception as repair_error:
                            repair_root = _root_cause(repair_error)
                            repair_message = str(repair_root)
                            repair_missing = _extract_missing_fields(repair_message) or missing_fields
                            Trace.event(
                                "judge_claims.repair_failed",
                                {
                                    "claim_id": frame.claim_id,
                                    "error": repair_message[:300],
                                },
                            )
                            return frame.claim_id, None, _build_error_payload(
                                error_type="llm_failed",
                                message=repair_message,
                                missing_fields=repair_missing,
                                repair_attempted=True,
                            )

                    return frame.claim_id, None, _build_error_payload(
                        error_type="llm_failed",
                        message=message,
                        missing_fields=missing_fields,
                    )

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
                "error_count": len(errors),
                "verdicts": {cid: out.verdict for cid, out in outputs.items()},
                "ui_locale": ui_locale,
            })
            Trace.event(
                "deep.claim_judged_count",
                {
                    "count": len(deep_ctx.claim_frames),
                    "ok": len(outputs),
                    "error": len(errors),
                },
            )

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("judge_claims")


class MarkJudgeUnavailableStep(Step):
    """Populate deep claim errors when judge capability is unavailable."""

    def __init__(self, reason: str = "judge_unavailable"):
        self._reason = reason

    weight: float = 1.0

    name: str = "judge_unavailable"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("judge_unavailable")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())
            if not deep_ctx.claim_frames:
                return ctx.set_extra("deep_claim_ctx", deep_ctx)

            errors = deep_ctx.errors or {}
            for frame in deep_ctx.claim_frames:
                if frame.claim_id in errors:
                    continue
                errors[frame.claim_id] = _build_error_payload(
                    error_type="judge_unavailable",
                    message=self._reason,
                )

            deep_ctx.errors = errors

            Trace.event(
                "judge_claims.unavailable",
                {"reason": self._reason, "claims": len(deep_ctx.claim_frames)},
            )

            return ctx.set_extra("deep_claim_ctx", deep_ctx)

        finally:
            Trace.phase_end("judge_unavailable")


class AssembleDeepResultStep(Step):
    """
    Step that assembles final deep analysis result.

    Produces per-claim outputs with no global scoring.
    """
    weight: float = 1.0

    name: str = "assemble_deep_result"

    def __init__(self, config: Any | None = None):
        self._config = config

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        Trace.phase_start("assemble_deep_result")

        try:
            deep_ctx: DeepClaimContext = ctx.extras.get("deep_claim_ctx", DeepClaimContext())

            # Use standardized AnalysisMode enum for API responses
            analysis_mode = ctx.mode.api_analysis_mode
            judge_mode = ScoringMode.DEEP.value

            claim_results: list[dict[str, Any]] = []
            claim_verdicts: list[dict[str, Any]] = []

            from spectrue_core.utils.trust_utils import enrich_sources_with_trust

            def _evidence_stats_payload(frame: ClaimFrame) -> dict[str, Any]:
                stats = frame.evidence_stats
                return {
                    "total_sources": stats.total_sources,
                    "support_sources": stats.support_sources,
                    "refute_sources": stats.refute_sources,
                    "context_sources": stats.context_sources,
                    "high_trust_sources": stats.high_trust_sources,
                    "direct_quotes": stats.direct_quotes,
                    "conflicting_evidence": stats.conflicting_evidence,
                    "missing_sources": stats.missing_sources,
                    "missing_direct_quotes": stats.missing_direct_quotes,
                    "exact_dupes_total": stats.exact_dupes_total,
                    "similar_clusters_total": stats.similar_clusters_total,
                    "publishers_total": stats.publishers_total,
                    "support": {
                        "precision_publishers": stats.support.precision_publishers,
                        "corroboration_clusters": stats.support.corroboration_clusters,
                    },
                    "refute": {
                        "precision_publishers": stats.refute.precision_publishers,
                        "corroboration_clusters": stats.refute.corroboration_clusters,
                    },
                }

            def _confirmation_payload(frame: ClaimFrame) -> dict[str, Any]:
                # Deep v2: use deterministic confirmation counts
                # Use value from runtime config if available
                from spectrue_core.runtime_config import DeepV2Config
                from spectrue_core.verification.scoring.confirmation_counts import compute_confirmation_counts
                
                runtime = getattr(self._config, "runtime", None)
                deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())
                lam = deep_v2_cfg.confirmation_lambda
                
                corr_meta = ctx.get_extra("corroboration_by_claim") or {}
                cid = frame.claim_id
                
                if isinstance(corr_meta, dict) and cid in corr_meta:
                    vals = compute_confirmation_counts(corr_meta[cid], lam=lam)
                    return {
                        "C_precise": vals["C_precise"],
                        "C_corr": vals["C_corr"],
                        "C_total": vals["C_total"],
                    }

                # Fallback to frame counts (if already calculated)
                counts = frame.confirmation_counts
                return {
                    "C_precise": counts.C_precise,
                    "C_corr": counts.C_corr,
                    "C_total": counts.C_total,
                }

            for frame in deep_ctx.claim_frames:
                judge_output = deep_ctx.judge_outputs.get(frame.claim_id)
                error = deep_ctx.errors.get(frame.claim_id)

                if error or judge_output is None:
                    error_payload = error or {"error_type": "judge_missing", "message": "Judge output missing"}
                    claim_result = {
                        "claim_id": frame.claim_id,
                        "status": "error",
                        "rgba": None,
                        "verdict_score": None,
                        "explanation": None,
                        "sources_used": [],
                        "error": error_payload,
                    }
                    if ctx.mode.api_analysis_mode == AnalysisMode.DEEP_V2:
                        claim_result["evidence_stats"] = _evidence_stats_payload(frame)
                        claim_result["confirmation_counts"] = _confirmation_payload(frame)
                    claim_results.append(claim_result)
                    continue

                rgba = [
                    judge_output.rgba.r,
                    judge_output.rgba.g,
                    judge_output.rgba.b,
                    judge_output.rgba.a,
                ]

                if rgba and len(rgba) == 4 and isinstance(rgba[3], (int, float)) and rgba[3] < 0:
                    stats_by_claim = ctx.get_extra("evidence_stats_by_claim") or {}
                    st = stats_by_claim.get(frame.claim_id) if isinstance(stats_by_claim, dict) else None
                    if isinstance(st, dict):
                        a_det = st.get("A_deterministic")
                        try:
                            a_det_f = float(a_det)
                        except Exception:
                            a_det_f = 0.0
                        rgba[3] = max(0.0, min(1.0, a_det_f))

                # M133: Alpha capping removed — LLM A-score passes through unchanged

                verdict_score = rgba[1] if isinstance(rgba, list) and len(rgba) > 1 else None
                sources_used_refs = list(judge_output.sources_used or [])

                # Build full sources list with trust info FIRST
                sources_list = []
                evidence_map = {ei.evidence_id: ei for ei in frame.evidence_items}
                url_map = {ei.url: ei for ei in frame.evidence_items if ei.url}

                for src_ref in sources_used_refs:
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

                # Enrich with trust categories for proper tier badges
                sources_list = enrich_sources_with_trust(sources_list)

                # Include FULL source objects in claim_results (not just refs)
                # This allows frontend to display proper tier badges
                claim_result = {
                    "claim_id": frame.claim_id,
                    "claim_text": frame.claim_text,
                    "status": "ok",
                    "rgba": rgba,
                    "verdict_score": verdict_score,
                    "explanation": judge_output.explanation,
                    "sources_used": sources_list,  # Full objects, not just refs
                }
                # Deep v2: attach corroboration counters (debug/UX optional)
                corr_meta = ctx.get_extra("corroboration_by_claim") or {}
                if isinstance(corr_meta, dict) and frame.claim_id in corr_meta:
                    claim_result["corroboration"] = corr_meta[frame.claim_id]

                if ctx.mode.api_analysis_mode == AnalysisMode.DEEP_V2:
                    claim_result["evidence_stats"] = _evidence_stats_payload(frame)
                    claim_result["confirmation_counts"] = _confirmation_payload(frame)
                claim_results.append(claim_result)

                # Clamp A score for display to avoid -1.0
                safe_rgba = list(rgba)
                if len(safe_rgba) == 4 and isinstance(safe_rgba[3], (int, float)):
                    safe_rgba[3] = max(0.0, min(1.0, float(safe_rgba[3])))

                claim_verdicts.append({
                    "claim_id": frame.claim_id,
                    "text": frame.claim_text,
                    "rgba": safe_rgba,
                    "verdict_score": verdict_score,
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

            rgba_audit_payload = None
            rgba_audit = ctx.get_extra(RGBA_AUDIT_KEY)
            if isinstance(rgba_audit, RGBAResult):
                rgba_audit_payload = RGBAAuditResultPayload.from_result(rgba_audit).to_payload()
            elif isinstance(rgba_audit, RGBAAuditResultPayload):
                rgba_audit_payload = rgba_audit.to_payload()
            elif isinstance(rgba_audit, dict):
                rgba_audit_payload = rgba_audit

            deep_analysis_payload = {
                "claim_results": claim_results,
            }
            if ctx.mode.api_analysis_mode == AnalysisMode.DEEP_V2:
                clusters_summary = ctx.get_extra("clusters_summary")
                if isinstance(clusters_summary, list):
                    deep_analysis_payload["clusters_summary"] = clusters_summary

            final_result = {
                "analysis_mode": analysis_mode,
                "judge_mode": judge_mode,
                "deep_analysis": deep_analysis_payload,
            }
            if rgba_audit_payload is not None:
                final_result["rgba_audit"] = rgba_audit_payload
            if deep_ctx.claim_frames:
                final_result["claims"] = [frame.claim_text for frame in deep_ctx.claim_frames]

            Trace.event("assemble_deep_result.complete", {"result_count": len(claim_results)})
            Trace.event(
                "final_result.keys",
                {
                    "judge_mode": final_result.get("judge_mode", ScoringMode.DEEP.value),
                    "keys": sorted(final_result.keys()),
                },
            )

            judgments = Judgments(standard=None, per_claim_results=tuple(claim_results))

            return (
                ctx.with_update(verdict=verdict)
                .set_extra("deep_claim_ctx", deep_ctx)
                .set_extra("deep_analysis_result", final_result.get("deep_analysis"))
                .set_extra("final_result", final_result)
                .set_extra(JUDGMENTS_KEY, judgments)
            )

        finally:
            Trace.phase_end("assemble_deep_result")


def get_deep_claim_steps(llm_client: LLMClient, config: Any | None = None) -> list[Step]:
    """
    Get all steps for deep claim processing.
    
    Args:
        llm_client: LLM client for skill calls
        config: Runtime configuration
    
    Returns:
        List of steps in execution order
    """
    return [
        BuildClaimFramesStep(config=config),
        SummarizeEvidenceStep(llm_client),
        JudgeClaimsStep(llm_client),
        AssembleDeepResultStep(config=config),
    ]
