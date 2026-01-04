# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Claim Judge skill for per-claim deep analysis.

Produces raw RGBA verdicts for individual claims without any postprocessing.
The output is returned directly to the frontend unchanged.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import CLAIM_JUDGE_SCHEMA
from spectrue_core.agents.skills.claim_judge_prompts import (
    build_claim_judge_prompt,
    build_claim_judge_system_prompt,
)
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    EvidenceSummary,
    JudgeOutput,
    RGBAScore,
)
from spectrue_core.utils.trace import Trace


class ClaimJudgeSkill:
    """
    Skill that produces RGBA verdicts for individual claims.
    
    The output of this skill is returned to the frontend without
    any modification (no postprocessing).
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize skill with LLM client.
        
        Args:
            llm_client: Configured LLM client for API calls
        """
        self.llm = llm_client

    async def judge(
        self,
        frame: ClaimFrame,
        evidence_summary: EvidenceSummary | None = None,
        *,
        ui_locale: str = "en",
    ) -> JudgeOutput:
        """
        Judge a claim and produce RGBA verdict.
        
        Args:
            frame: ClaimFrame with claim and evidence
            evidence_summary: Optional pre-analyzed evidence summary
            ui_locale: UI language code for explanation output (e.g., "uk", "en")
        
        Returns:
            JudgeOutput with RGBA scores and verdict (unchanged from LLM)
        """
        # Build prompt with UI locale for explanation language
        user_prompt = build_claim_judge_prompt(frame, evidence_summary, ui_locale=ui_locale)
        system_prompt = build_claim_judge_system_prompt(lang=ui_locale)

        Trace.event("claim_judge.start", {
            "claim_id": frame.claim_id,
            "evidence_count": len(frame.evidence_items),
            "has_summary": evidence_summary is not None,
        })

        try:
            # Call LLM with structured output
            response = await self.llm.call_structured(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CLAIM_JUDGE_SCHEMA,
                schema_name="claim_judge",
            )

            # Parse response into JudgeOutput (no modifications)
            judge_output = self._parse_response(response, frame)

            # Validate sources_used constraint
            judge_output = self._validate_sources_used(judge_output, frame)

            Trace.event("claim_judge.complete", {
                "claim_id": frame.claim_id,
                "verdict": judge_output.verdict,
                "confidence": judge_output.confidence,
                "rgba": judge_output.rgba.to_dict(),
            })

            return judge_output

        except Exception as e:
            Trace.event("claim_judge.error", {
                "claim_id": frame.claim_id,
                "error": str(e),
            })

            raise RuntimeError(f"Claim judge failed: {e}") from e

    def _parse_response(
        self,
        response: dict[str, Any],
        frame: ClaimFrame,
    ) -> JudgeOutput:
        """Parse LLM response into JudgeOutput without modification."""

        # Extract RGBA scores
        rgba_dict = response.get("rgba", {})
        rgba = RGBAScore(
            r=float(rgba_dict.get("R", 0.0)),
            g=float(rgba_dict.get("G", 0.5)),
            b=float(rgba_dict.get("B", 0.5)),
            a=float(rgba_dict.get("A", 0.3)),
        )

        # Extract other fields
        confidence = float(response.get("confidence", 0.3))
        verdict = str(response.get("verdict", "NEI"))
        explanation = str(response.get("explanation", ""))

        # Extract sources_used and missing_evidence
        sources_used = tuple(str(s) for s in response.get("sources_used", []))
        missing_evidence = tuple(str(m) for m in response.get("missing_evidence", []))

        return JudgeOutput(
            claim_id=frame.claim_id,
            rgba=rgba,
            confidence=confidence,
            verdict=verdict,
            explanation=explanation,
            sources_used=sources_used,
            missing_evidence=missing_evidence,
        )

    def _validate_sources_used(
        self,
        output: JudgeOutput,
        frame: ClaimFrame,
    ) -> JudgeOutput:
        """
        Validate that sources_used only contains URLs from evidence_items.
        
        This is the ONLY modification allowed - filtering invalid URLs.
        The validation is required by FR-009.
        """
        available_urls = frozenset(item.url for item in frame.evidence_items)

        valid_sources = tuple(
            url for url in output.sources_used
            if url in available_urls
        )

        if len(valid_sources) != len(output.sources_used):
            Trace.event("claim_judge.invalid_sources_filtered", {
                "claim_id": frame.claim_id,
                "original_count": len(output.sources_used),
                "valid_count": len(valid_sources),
            })

        # Only update if different
        if valid_sources != output.sources_used:
            return JudgeOutput(
                claim_id=output.claim_id,
                rgba=output.rgba,
                confidence=output.confidence,
                verdict=output.verdict,
                explanation=output.explanation,
                sources_used=valid_sources,
                missing_evidence=output.missing_evidence,
            )

        return output

    def _fallback_output(self, frame: ClaimFrame, error: str) -> JudgeOutput:
        """
        Deprecated: fallback outputs are not permitted in deep mode.
        """
        raise RuntimeError(f"Claim judge failed: {error}")
