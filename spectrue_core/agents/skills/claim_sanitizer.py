# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Claim Sanitization Skill (LLM-based).
Removes actionable medical instructions from user-facing text.
"""

from __future__ import annotations

import logging
import hashlib
from typing import TypedDict

from .base_skill import BaseSkill

logger = logging.getLogger(__name__)


class SanitizedClaimResult(TypedDict):
    """Result of claim sanitization."""
    text_safe: str
    is_actionable_medical: bool
    danger_tags: list[str]
    redacted_spans: list[dict[str, int | str]]


class ClaimSanitizerSkill(BaseSkill):
    """
    Sanitizes extracted claims to remove actionable medical instructions.
    
    This skill ensures that unsafe content (doses, step-by-step procedures) 
    is redacted from logs, UI, and exports, while preserving the raw claim 
    text internally for verification purposes.
    
    Fail-closed behavior:
    - If LLM fails or times out, the claim is marked as actionable medical 
      and the text is fully redacted to prevent leak.
    """

    async def sanitize_claim(
        self,
        text: str,
        kind: str = "core",
        topic_key: str = "default",
    ) -> SanitizedClaimResult:
        """
        Sanitize a single claim.
        
        Args:
            text: Raw claim text
            kind: Claim kind (e.g., 'medical_claim', 'core')
            topic_key: Topic identifier
            
        Returns:
            SanitizedClaimResult with text_safe and safety flags
        """
        if not text:
            return {
                "text_safe": "",
                "is_actionable_medical": False,
                "danger_tags": [],
                "redacted_spans": [],
            }

        # Cache key construction
        content_hash = hashlib.sha256(
            f"{text}|{kind}|{topic_key}|v1".encode()
        ).hexdigest()
        
        instructions = self._get_instructions()
        prompt = self._build_prompt(text, kind, topic_key)
        
        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                timeout=10.0,
                cache_key=f"sanitize_v1_{content_hash}",
                trace_kind="claim_sanitizer",
            )
            
            return self._parse_result(result, text)
            
        except Exception as e:
            logger.warning("[M74] Claim sanitization failed: %s. Fail-closed.", e)
            return self._fail_closed_result(text)

    def _get_instructions(self) -> str:
        return """You are a Safety Sanitizer for a fact-checking system.
Your goal is to REMOVE actionable medical instructions from claims while preserving their propositional meaning for verification context.

## Rules
1. **PRESERVE Meaning**: "Kerosene treats cancer" must remain (it is a verifiable claim).
2. **REDACT Actionable Details**: Use `[REDACTED_MEDICAL]` for:
   - Doses (500mg, 10ml, 3 drops)
   - Frequencies (3 times a day, every 4 hours)
   - Procedures (mix with sugar, drink on empty stomach, inject)
   - Durations (for 3 weeks)
3. **NO New Facts**: Do not rewrite the claim to change its meaning.
4. **Strict JSON**: Output only valid JSON.

## Examples
Input: "To cure cancer, take 1 tsp of kerosene daily for 2 weeks."
Output:
{
  "text_safe": "To cure cancer, take [REDACTED_MEDICAL] of kerosene [REDACTED_MEDICAL] for [REDACTED_MEDICAL].",
  "is_actionable_medical": true,
  "danger_tags": ["medical_instruction", "dosing"]
}

Input: "Kerosene is claimed to be a cure for diabetes."
Output:
{
  "text_safe": "Kerosene is claimed to be a cure for diabetes.",
  "is_actionable_medical": false,
  "danger_tags": []
}"""

    def _build_prompt(self, text: str, kind: str, topic_key: str) -> str:
        return f"""Sanitize this claim.

Claim: "{text}"
Kind: {kind}
Topic: {topic_key}

Return JSON with keys: text_safe, is_actionable_medical, danger_tags, redacted_spans.
"""

    def _parse_result(self, result: dict, original_text: str) -> SanitizedClaimResult:
        try:
            text_safe = str(result.get("text_safe", ""))
            
            # Integrity check: text_safe should not be empty if original wasn't
            if not text_safe and original_text:
                return self._fail_closed_result(original_text)
                
            is_med = bool(result.get("is_actionable_medical", False))
            tags = [str(t) for t in result.get("danger_tags", [])]
            
            # Parse spans if provided, else empty
            spans = []
            raw_spans = result.get("redacted_spans", [])
            if isinstance(raw_spans, list):
                for s in raw_spans:
                    if isinstance(s, dict) and "start" in s and "end" in s:
                        spans.append(s)
            
            if "medical_instruction" in tags:
                is_med = True
                
            return {
                "text_safe": text_safe,
                "is_actionable_medical": is_med,
                "danger_tags": tags,
                "redacted_spans": spans,
            }
            
        except Exception:
            return self._fail_closed_result(original_text)

    def _fail_closed_result(self, original_text: str) -> SanitizedClaimResult:
        """
        Fail-closed fallback. 
        If we can't sanitize, we assume it's dangerous action and redact everything.
        This effectively hides the claim text from UI/Logs but keeps it in the system.
        """
        return {
            "text_safe": "[REDACTED_MEDICAL_CONTENT]",
            "is_actionable_medical": True,
            "danger_tags": ["sanitization_failed", "fail_closed"],
            "redacted_spans": [{"start": 0, "end": len(original_text), "label": "fail_closed"}]
        }
