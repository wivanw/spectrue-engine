# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Evidence Summarizer skill for per-claim deep analysis.

Categorizes evidence items by their stance relative to the claim
and identifies evidence gaps.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import EVIDENCE_SUMMARIZER_SCHEMA
from spectrue_core.agents.skills.evidence_summarizer_prompts import (
    build_evidence_summarizer_prompt,
    build_evidence_summarizer_system_prompt,
)
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    EvidenceReference,
    EvidenceSummary,
)
from spectrue_core.utils.trace import Trace


class EvidenceSummarizerSkill:
    """
    Skill that categorizes evidence for a claim.
    
    Uses LLM to analyze evidence items and group them by stance
    (supporting, refuting, contextual) while identifying gaps.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize skill with LLM client.
        
        Args:
            llm_client: Configured LLM client for API calls
        """
        self.llm = llm_client

    async def summarize(self, frame: ClaimFrame) -> EvidenceSummary:
        """
        Summarize evidence for a claim.
        
        Args:
            frame: ClaimFrame with evidence to analyze
        
        Returns:
            EvidenceSummary with categorized evidence
        """
        # Skip if no evidence
        if not frame.evidence_items:
            Trace.event("evidence_summarizer.skip", {
                "claim_id": frame.claim_id,
                "reason": "no_evidence",
            })
            return EvidenceSummary(
                evidence_gaps=("No sources found for this claim",),
                conflicts_present=False,
            )

        # Build prompt
        user_prompt = build_evidence_summarizer_prompt(frame)
        system_prompt = build_evidence_summarizer_system_prompt()

        Trace.event("evidence_summarizer.start", {
            "claim_id": frame.claim_id,
            "evidence_count": len(frame.evidence_items),
        })

        try:
            # Call LLM with structured output
            response = await self.llm.call_structured(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=EVIDENCE_SUMMARIZER_SCHEMA,
                schema_name="evidence_summarizer",
            )

            # Parse response
            summary = self._parse_response(response, frame.claim_id)

            Trace.event("evidence_summarizer.complete", {
                "claim_id": frame.claim_id,
                "supporting": len(summary.supporting_evidence),
                "refuting": len(summary.refuting_evidence),
                "contextual": len(summary.contextual_evidence),
                "conflicts": summary.conflicts_present,
            })

            return summary

        except Exception as e:
            Trace.event("evidence_summarizer.error", {
                "claim_id": frame.claim_id,
                "error": str(e),
            })

            # Return degraded summary on error
            return self._fallback_summary(frame)

    def _parse_response(
        self, 
        response: dict[str, Any], 
        claim_id: str
    ) -> EvidenceSummary:
        """Parse LLM response into EvidenceSummary."""

        def parse_refs(items: list[dict]) -> tuple[EvidenceReference, ...]:
            refs: list[EvidenceReference] = []
            for item in items:
                if isinstance(item, dict):
                    refs.append(EvidenceReference(
                        evidence_id=str(item.get("evidence_id", "")),
                        reason=str(item.get("reason", "")),
                    ))
            return tuple(refs)

        supporting = parse_refs(response.get("supporting_evidence", []))
        refuting = parse_refs(response.get("refuting_evidence", []))
        contextual = parse_refs(response.get("contextual_evidence", []))

        gaps = tuple(str(g) for g in response.get("evidence_gaps", []))
        conflicts = bool(response.get("conflicts_present", False))

        return EvidenceSummary(
            supporting_evidence=supporting,
            refuting_evidence=refuting,
            contextual_evidence=contextual,
            evidence_gaps=gaps,
            conflicts_present=conflicts,
        )

    def _fallback_summary(self, frame: ClaimFrame) -> EvidenceSummary:
        """
        Create fallback summary when LLM fails.
        
        Uses stance from evidence items if available.
        """
        supporting: list[EvidenceReference] = []
        refuting: list[EvidenceReference] = []
        contextual: list[EvidenceReference] = []

        for item in frame.evidence_items:
            ref = EvidenceReference(
                evidence_id=item.evidence_id,
                reason="Categorized from initial stance",
            )

            stance = (item.stance or "").upper()
            if stance == "SUPPORT":
                supporting.append(ref)
            elif stance == "REFUTE":
                refuting.append(ref)
            else:
                contextual.append(ref)

        return EvidenceSummary(
            supporting_evidence=tuple(supporting),
            refuting_evidence=tuple(refuting),
            contextual_evidence=tuple(contextual),
            evidence_gaps=("Unable to analyze gaps due to error",),
            conflicts_present=len(supporting) > 0 and len(refuting) > 0,
        )