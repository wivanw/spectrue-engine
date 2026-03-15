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

from spectrue_core.llm.llm_client import LLMClient
from spectrue_core.use_cases.claims.deep_judge import (
    summarize_evidence_for_claims,
    judge_claims_independently,
    _build_error_payload,
)
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
from spectrue_core.pipeline.claims.claim_frame_builder import (
    build_claim_frames_from_contexts,
)
from spectrue_core.pipeline.claims.execution_context import ClaimExecutionContext
from spectrue_core.adapters.llm.evidence_summarizer import EvidenceSummarizerSkill


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

            # Lookup or create claim contexts
            claim_contexts: dict[str, ClaimExecutionContext] = ctx.extras.get("claim_contexts", {})
            
            if not claim_contexts and claims:
                for claim in claims:
                    cid = claim.get("id") or claim.get("claim_id") or "unknown"
                    if cid not in claim_contexts:
                        st = execution_states.get(cid)
                        evs = evidence_by_claim.get(cid, [])
                        claim_contexts[cid] = ClaimExecutionContext.create(
                            claim=claim,
                            evidence_items=evs,
                            state=st
                        )
                # Persist context backwards to pipeline
                ctx = ctx.set_extra("claim_contexts", claim_contexts)

            if not claim_contexts:
                Trace.event("build_claim_frames.skip", {"reason": "no_claims"})
                return ctx.set_extra("deep_claim_ctx", DeepClaimContext())

            confirmation_lambda = None
            if ctx.mode.api_analysis_mode == AnalysisMode.DEEP_V2:
                from spectrue_core.runtime_config import DeepV2Config
                runtime = getattr(self._config, "runtime", None)
                deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())
                confirmation_lambda = deep_v2_cfg.confirmation_lambda

            # Build frames from contexts (T005)
            frames = build_claim_frames_from_contexts(
                claim_contexts=claim_contexts,
                document_text=document_text,
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
    weight: float = 9.0  # ~9s actual

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

            # Pre-summarization cleaning (US3)
            skill = EvidenceSummarizerSkill(self._llm)
            
            # Use max_doc_concurrency for parallel cleaning
            max_doc_conc = 4
            max_claim_conc = 4
            
            runtime = ctx.get_extra("runtime_config")
            if runtime and hasattr(runtime, "llm"):
                max_doc_conc = getattr(runtime.llm, "max_doc_concurrency", 4)
                max_claim_conc = getattr(runtime.llm, "max_claim_concurrency", 4)

            # Parallel cleaning of evidence documents (US2.2)
            cleaned_tasks = [skill.clean_evidence_for_frame(f, max_concurrency=max_doc_conc) for f in deep_ctx.claim_frames]
            cleaned_frames = await asyncio.gather(*cleaned_tasks)
            deep_ctx.claim_frames = cleaned_frames

            from spectrue_core.llm.model_registry import ModelID

            summaries = await summarize_evidence_for_claims(
                claim_frames=deep_ctx.claim_frames,
                llm_client=self._llm,
                progress_callback=ctx.get_extra("progress_callback"),
                max_concurrency=max_claim_conc,
                model=ModelID.NANO,
            )

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
    
    Uses judge_claims_independently use case to generate RGBA scores and verdicts.
    Output is returned unchanged to the frontend.
    """
    weight: float = 25.0  # ~25s actual (LLM judging per claim)

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

            # Get UI locale from pipeline context
            # This is the user's interface language from the API request
            ui_locale = ctx.lang or "en"
            analysis_mode = ctx.mode.api_analysis_mode

            # Use max_claim_concurrency for independent judging
            max_claim_conc = 4
            runtime = ctx.get_extra("runtime_config")
            if runtime and hasattr(runtime, "llm"):
                max_claim_conc = getattr(runtime.llm, "max_claim_concurrency", 4)

            outputs, errors = await judge_claims_independently(
                claim_frames=deep_ctx.claim_frames,
                evidence_summaries=deep_ctx.evidence_summaries,
                llm_client=self._llm,
                ui_locale=ui_locale,
                analysis_mode=analysis_mode,
                progress_callback=ctx.get_extra("progress_callback"),
                max_concurrency=max_claim_conc,
            )

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

                # Deep mode uses LLM returned RGBA directly without deterministic A overrides.

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
                # Corroboration UX is driven by frame.confirmation_counts payload

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
                # Serialize claim graph for report (compact: numeric enum codes)
                graph_result = ctx.get_extra("graph_result")
                claim_id_to_text = {
                    r["claim_id"]: r.get("claim_text") or r.get("text") or ""
                    for r in claim_results
                }
                claim_id_to_rgba = {
                    r["claim_id"]: r["rgba"]
                    for r in claim_results
                    if r.get("rgba") is not None
                }
                claim_id_to_type: dict[str, str] = {}
                for i, c in enumerate(ctx.claims or []):
                    cid = c.get("id") or c.get("claim_id") or f"c{i + 1}"
                    claim_id_to_type[cid] = c.get("type", "core")
                if graph_result is not None and not getattr(graph_result, "disabled", True):
                    from spectrue_core.domain.claims.graph.report_serializer import (
                        serialize_graph_for_report,
                    )
                    deep_analysis_payload["claim_graph"] = serialize_graph_for_report(
                        graph_result,
                        claim_id_to_text,
                        claim_id_to_rgba=claim_id_to_rgba,
                        claim_id_to_type=claim_id_to_type,
                    )
                elif claim_results:
                    # Fallback: minimal graph (nodes only) when ClaimGraphStep was skipped
                    # so the 3D tree button and shared report tree still work
                    from spectrue_core.domain.claims.graph.report_serializer import (
                        _claim_type_to_code,
                        fallback_edges_for_nodes,
                    )
                    nodes = []
                    for i, r in enumerate(claim_results):
                        cid = r.get("claim_id") or f"c{i + 1}"
                        ct = claim_id_to_type.get(cid, "core")
                        rgba = claim_id_to_rgba.get(cid)
                        node: dict = {
                            "claim_id": cid,
                            "text": claim_id_to_text.get(cid, ""),
                            "pre_meta": {},
                            "post_meta": {},
                            "centrality": 0.0,
                            "is_key_claim": 1 if i == 0 else 0,
                            "in_structural_weight": 0.0,
                            "in_contradict_weight": 0.0,
                            "claim_type": _claim_type_to_code(ct),
                        }
                        if rgba is not None and len(rgba) >= 4:
                            node["rgba"] = [round(float(x), 4) for x in rgba[:4]]
                        nodes.append(node)
                    deep_analysis_payload["claim_graph"] = {
                        "nodes": nodes,
                        "edges": fallback_edges_for_nodes(nodes),
                    }

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
