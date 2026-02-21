"""Use cases for deep claim judging and evidence summarization."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from spectrue_core.adapters.llm.claim_judge import ClaimJudgeSkill
from spectrue_core.adapters.llm.evidence_summarizer import EvidenceSummarizerSkill
from spectrue_core.adapters.llm.claim_judge_prompts import (
    build_claim_judge_prompt,
    build_claim_judge_system_prompt,
)
from spectrue_core.domain.claims.frame import ClaimFrame, EvidenceSummary, JudgeOutput
from spectrue_core.adapters.llm.scoring_contract import CLAIM_JUDGE_SCHEMA
from spectrue_core.llm.model_registry import ModelID
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


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


_SCHEMA_MISSING_RE = re.compile(r"\$\.(?P<field>[A-Za-z0-9_\\[\\].]+): missing required field")


def _extract_missing_fields(message: str) -> list[str]:
    if not message:
        return []
    return list({match.group("field") for match in _SCHEMA_MISSING_RE.finditer(message)})


def _is_format_error(exc: Exception) -> bool:
    from spectrue_core.llm.failures import is_schema_failure
    if is_schema_failure(exc):
        return True
    msg = str(exc).lower()
    return "json parse" in msg or "invalid json" in msg


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


def build_judge_evidence_stats(frame: ClaimFrame) -> dict[str, Any]:
    """
    Build a compact, structured evidence_stats object for the judge.
    This is purely observational/diagnostic:
    - it must NOT be used as a substitute for evidence spans
    - it helps the judge reason about redundancy vs independence and coverage
    """
    out: dict[str, Any] = {}
    st = frame.evidence_stats
    if st:
        out["sources_observed"] = st.total_sources
        out["unique_urls"] = st.total_sources
        out["unique_domains"] = st.publishers_total
        out["direct_anchors"] = st.direct_quotes
        out["covered_slots"] = 0
        out["transferred"] = 0
        out["precision_publishers_support"] = st.support.precision_publishers
        out["precision_publishers_refute"] = st.refute.precision_publishers
        out["corroboration_clusters_support"] = st.support.corroboration_clusters
        out["corroboration_clusters_refute"] = st.refute.corroboration_clusters
        out["unique_publishers_total"] = st.publishers_total
        out["exact_content_groups"] = st.exact_dupes_total

    return out


async def summarize_evidence_for_claims(
    *,
    claim_frames: list[ClaimFrame],
    llm_client: Any,
    progress_callback: Any | None = None,
) -> dict[str, EvidenceSummary]:
    """Summarize evidence for each claim in parallel."""
    if not claim_frames:
        return {}

    skill = EvidenceSummarizerSkill(llm_client)

    processed = 0
    total = len(claim_frames)

    async def summarize_one(frame: ClaimFrame) -> tuple[str, EvidenceSummary]:
        nonlocal processed
        
        summary = await skill.summarize(frame)
        
        processed += 1
        if progress_callback:
            try:
                await progress_callback("analyzing_sentences", processed=processed, total=total)
            except Exception:
                pass
        return frame.claim_id, summary

    tasks = [summarize_one(frame) for frame in claim_frames]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries: dict[str, EvidenceSummary] = {}
    for result in results:
        if isinstance(result, Exception):
            Trace.event("summarize_evidence.task_error", {"error": str(result)})
            continue
        claim_id, summary = result
        summaries[claim_id] = summary

    return summaries


async def judge_claims_independently(
    *,
    claim_frames: list[ClaimFrame],
    evidence_summaries: dict[str, EvidenceSummary],
    llm_client: Any,
    ui_locale: str = "en",
    analysis_mode: Any = "general",
    progress_callback: Any | None = None,
) -> tuple[dict[str, JudgeOutput], dict[str, dict[str, Any]]]:
    """Judge claims independently in parallel with repair logic."""
    if not claim_frames:
        return {}, {}

    processed = 0
    total = len(claim_frames)
    
    skill = ClaimJudgeSkill(llm_client)
    
    # Internal helper for repair logic
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
            "sources_used, missing_evidence, prior_score, prior_reason.\n\n"
            f"{base_prompt}"
        )
        repair_system = build_claim_judge_system_prompt(lang=ui_locale)
        repair_system = f"{repair_system}\nReturn only JSON; no markdown or extra text."

        response = await llm_client.call_json(
            model=llm_client.model or ModelID.NANO,
            input=repair_prompt,
            instructions=repair_system,
            response_schema=CLAIM_JUDGE_SCHEMA,
            reasoning_effort="low",
            trace_kind="claim_judge.repair",
        )

        repaired = skill._parse_response(response, frame)
        return skill._validate_sources_used(repaired, frame)

    async def judge_one(frame: ClaimFrame) -> tuple[str, JudgeOutput | None, dict[str, Any] | None]:
        summary = evidence_summaries.get(frame.claim_id)
        try:
            evidence_stats = build_judge_evidence_stats(frame)

            output = await skill.judge(
                frame,
                summary,
                ui_locale=ui_locale,
                analysis_mode=analysis_mode,
                evidence_stats=evidence_stats,
            )
            
            # US3: Explicit confidence penalty for evidence-insufficient cleaned payloads
            if frame.evidence_items:
                total_ev = len(frame.evidence_items)
                boilerplate_count = sum(1 for e in frame.evidence_items if e.cleanliness and e.cleanliness.is_boilerplate)
                if boilerplate_count == total_ev and total_ev > 0:
                    import dataclasses
                    # If ALL evidence is boilerplate, penalize confidence heavily
                    new_conf = max(0.0, output.confidence - 0.5)
                    output = dataclasses.replace(output, confidence=new_conf)

            # US4: Connect prior_score, missing_evidence, and freshness adjustments to deep confidence/verdict composition
            original_conf = output.confidence
            original_verdict = output.verdict
            new_conf = original_conf
            new_verdict = original_verdict
            reason_codes = []

            # 1. Use missing_evidence to cap confidence if high
            if getattr(output, "missing_evidence", None) and len(output.missing_evidence) > 0:
                if new_conf > 0.5:
                    new_conf = 0.5
                    reason_codes.append("missing_evidence_penalty")

            # 2. Shift default "unverifiable" towards prior if prior is strong
            if output.verdict.lower() in ("unverified", "nei", "unverifiable") or output.rgba.g == -1.0:
                if getattr(output, "prior_score", -1.0) >= 0.8:
                    new_conf = max(0.4, min(new_conf + 0.3, 0.6))
                    new_verdict = "supported"
                    reason_codes.append("prior_score_supported_shift")
                elif getattr(output, "prior_score", -1.0) >= 0.0 and getattr(output, "prior_score", -1.0) <= 0.2:
                    new_conf = max(0.4, min(new_conf + 0.3, 0.6))
                    new_verdict = "refuted"
                    reason_codes.append("prior_score_refuted_shift")

            # 3. Apply FreshnessSignal modifier
            try:
                from spectrue_core.use_cases.verification.scoring.freshness_signal import calculate_freshness_adjustment
                freshness_adj = calculate_freshness_adjustment(frame.claim_id, frame.evidence_items)
                if freshness_adj < 0:
                    new_conf = max(0.0, new_conf + freshness_adj)
                    reason_codes.append("freshness_penalty")
            except ImportError:
                pass # freshness_signal module may not be available yet

            import dataclasses
            output = dataclasses.replace(
                output, 
                confidence=new_conf, 
                verdict=new_verdict
            )

            # Emit decision-impact event
            if reason_codes:
                Trace.event("decision_impact", {
                    "claim_id": frame.claim_id,
                    "module_name": "deep_judge_composition",
                    "did_change_retrieval": False,
                    "did_change_confidence": new_conf != original_conf,
                    "did_change_verdict": new_verdict != original_verdict,
                    "reason_codes": reason_codes,
                    "before_snapshot": {"confidence": original_conf, "verdict": original_verdict},
                    "after_snapshot": {"confidence": new_conf, "verdict": new_verdict},
                })

            processed += 1
            if progress_callback:
                try:
                    await progress_callback("analyzing_sentences", processed=processed, total=total)
                except Exception:
                    pass
            return frame.claim_id, output, None
        except Exception as e:
            root = _root_cause(e)
            message = str(root)
            missing_fields = _extract_missing_fields(message)

            if _is_format_error(root):
                try:
                    repaired = await _repair_claim_output(frame, summary)
                    processed += 1
                    if progress_callback:
                        try:
                            await progress_callback("analyzing_sentences", processed=processed, total=total)
                        except Exception:
                            pass
                    return frame.claim_id, repaired, None
                except Exception as repair_error:
                    repair_root = _root_cause(repair_error)
                    repair_message = str(repair_root)
                    repair_missing = _extract_missing_fields(repair_message) or missing_fields
                    processed += 1
                    if progress_callback:
                        try:
                            await progress_callback("analyzing_sentences", processed=processed, total=total)
                        except Exception:
                            pass
                    return frame.claim_id, None, _build_error_payload(
                        error_type="llm_failed",
                        message=repair_message,
                        missing_fields=repair_missing,
                        repair_attempted=True,
                    )

            processed += 1
            if progress_callback:
                try:
                    await progress_callback("analyzing_sentences", processed=processed, total=total)
                except Exception:
                    pass
            return frame.claim_id, None, _build_error_payload(
                error_type="llm_failed",
                message=message,
                missing_fields=missing_fields,
            )

    tasks = [judge_one(frame) for frame in claim_frames]
    results = await asyncio.gather(*tasks)

    outputs: dict[str, JudgeOutput] = {}
    errors: dict[str, dict[str, Any]] = {}
    for claim_id, output, error in results:
        if error:
            errors[claim_id] = error
            continue
        if output:
            outputs[claim_id] = output

    return outputs, errors