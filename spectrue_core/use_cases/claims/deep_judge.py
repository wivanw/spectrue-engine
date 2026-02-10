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


def build_judge_evidence_stats(
    *,
    claim_id: str,
    corroboration_by_claim: dict[str, dict[str, Any]] | None,
    evidence_stats_by_claim: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Build a compact, structured evidence_stats object for the judge.
    This is purely observational/diagnostic:
    - it must NOT be used as a substitute for evidence spans
    - it helps the judge reason about redundancy vs independence and coverage
    """
    out: dict[str, Any] = {}

    corr = (corroboration_by_claim or {}).get(claim_id) if isinstance(corroboration_by_claim, dict) else None
    est = (evidence_stats_by_claim or {}).get(claim_id) if isinstance(evidence_stats_by_claim, dict) else None

    if isinstance(est, dict):
        out["sources_observed"] = est.get("sources_observed", 0)
        out["unique_urls"] = est.get("unique_urls", 0)
        out["unique_domains"] = est.get("unique_domains", 0)
        out["direct_anchors"] = est.get("direct_anchors", 0)
        out["covered_slots"] = est.get("covered_slots", 0)
        out["transferred"] = est.get("transferred", 0)

    if isinstance(corr, dict):
        out["precision_publishers_support"] = corr.get("precision_publishers_support", 0)
        out["precision_publishers_refute"] = corr.get("precision_publishers_refute", 0)
        out["corroboration_clusters_support"] = corr.get("corroboration_clusters_support", 0)
        out["corroboration_clusters_refute"] = corr.get("corroboration_clusters_refute", 0)
        out["unique_publishers_total"] = corr.get("unique_publishers_total", 0)
        out["exact_content_groups"] = corr.get("exact_content_groups", 0)

    return out


async def summarize_evidence_for_claims(
    *,
    claim_frames: list[ClaimFrame],
    llm_client: Any,
) -> dict[str, EvidenceSummary]:
    """Summarize evidence for each claim in parallel."""
    if not claim_frames:
        return {}

    skill = EvidenceSummarizerSkill(llm_client)

    async def summarize_one(frame: ClaimFrame) -> tuple[str, EvidenceSummary]:
        summary = await skill.summarize(frame)
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
    evidence_stats_by_claim: dict[str, Any] | None = None,
    corroboration_by_claim: dict[str, Any] | None = None,
) -> tuple[dict[str, JudgeOutput], dict[str, dict[str, Any]]]:
    """Judge claims independently in parallel with repair logic."""
    if not claim_frames:
        return {}, {}

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
            "sources_used, missing_evidence.\n\n"
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
            evidence_stats = build_judge_evidence_stats(
                claim_id=frame.claim_id,
                corroboration_by_claim=corroboration_by_claim,
                evidence_stats_by_claim=evidence_stats_by_claim,
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
                try:
                    repaired = await _repair_claim_output(frame, summary)
                    return frame.claim_id, repaired, None
                except Exception as repair_error:
                    repair_root = _root_cause(repair_error)
                    repair_message = str(repair_root)
                    repair_missing = _extract_missing_fields(repair_message) or missing_fields
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