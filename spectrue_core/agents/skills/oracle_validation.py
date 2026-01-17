# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Oracle Validation Skill.

Provides semantic validation of Google Fact Check API results against user claims.
Uses LLM to compute relevance_score (0-1) instead of binary yes/no.
"""

from spectrue_core.agents.skills.base_skill import BaseSkill
from spectrue_core.verification.evidence.evidence_pack import OracleStatus
from spectrue_core.utils.trace import Trace
from spectrue_core.llm.model_registry import ModelID
import logging

logger = logging.getLogger(__name__)


# Threshold constants for hybrid Oracle flow
JACKPOT_THRESHOLD = 0.9   # Above this: STOP pipeline, return Oracle result
EVIDENCE_THRESHOLD = 0.5  # Above this (but below JACKPOT): add to evidence pack


class OracleValidationSkill(BaseSkill):
    """
    LLM-based semantic validation for Oracle (Google Fact Check) results.
    
    Instead of binary "relevant/not relevant", computes a relevance_score
    that enables three-scenario hybrid flow:
    - JACKPOT (>0.9): Exact match, stop pipeline
    - EVIDENCE (0.5-0.9): Related, add to pack
    - MISS (<0.5): Not relevant, ignore
    """

    async def validate(
        self, 
        user_claim: str, 
        oracle_claim: str, 
        oracle_rating: str,
        oracle_summary: str = ""
    ) -> dict:
        """
        Validate semantic relevance between user claim and Oracle result.
        
        Args:
            user_claim: The original claim from user's text
            oracle_claim: The claim text from fact-check database
            oracle_rating: The verdict/rating from fact-checker (e.g., "False", "Mostly True")
            oracle_summary: Optional explanation/summary from fact-checker
            
        Returns:
            {
                "relevance_score": float (0.0-1.0),
                "status": "CONFIRMED" | "REFUTED" | "MIXED",
                "reasoning": str,
                "is_jackpot": bool
            }
        """
        if not user_claim or not oracle_claim:
            return self._empty_result("Missing claim text")

        prompt = self._build_prompt(user_claim, oracle_claim, oracle_rating, oracle_summary)

        try:
            result = await self.llm_client.call_json(
                model=ModelID.NANO,
                input=prompt,
                instructions=self._get_instructions(),
                reasoning_effort="low",
                timeout=15.0,
                trace_kind="oracle_validation"
            )

            relevance_score = float(result.get("relevance_score", 0.0))
            relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to 0-1

            status = self._map_status(result.get("status", ""))
            is_jackpot = relevance_score > JACKPOT_THRESHOLD

            # Extract LLM-determined scores (no heuristics!)
            verified_score = float(result.get("verified_score", -1.0))
            danger_score = float(result.get("danger_score", -1.0))

            # Warning only - no fallback! If -1, it's a bug and should be visible.
            if verified_score < 0 or danger_score < 0:
                logger.warning("[OracleValidation] ⚠️ LLM did not return verified_score/danger_score. BUG!")

            validated_result = {
                "relevance_score": relevance_score,
                "status": status,
                "verified_score": verified_score,
                "danger_score": danger_score,
                "reasoning": result.get("reasoning", ""),
                "is_jackpot": is_jackpot
            }

            Trace.event("oracle.validation", {
                "user_claim": user_claim[:80],
                "oracle_claim": oracle_claim[:80],
                "relevance_score": relevance_score,
                "verified_score": verified_score,
                "danger_score": danger_score,
                "status": status,
                "is_jackpot": is_jackpot
            })

            log_level = logging.INFO if is_jackpot else logging.DEBUG
            logger.log(log_level, 
                "[OracleValidation] score=%.2f, v=%.2f, status=%s, jackpot=%s", 
                relevance_score, verified_score, status, is_jackpot
            )

            return validated_result

        except Exception as e:
            logger.warning("[OracleValidation] LLM call failed: %s. Returning MISS.", e)
            return self._empty_result(f"LLM error: {e}")

    async def validate_batch(
        self,
        user_claim: str,
        candidates: list[dict],
    ) -> dict:
        """
        Batch validate ALL candidates in a SINGLE LLM call.
        
        This is more efficient than N separate calls and allows LLM
        to compare candidates against each other.
        
        Args:
            user_claim: The original claim from user's text
            candidates: List of candidate dicts from Google Fact Check API
                Each has: claim_text, rating, title, publisher, url, index
            
        Returns:
            {
                "best_index": int (-1 if no good match),
                "relevance_score": float (0.0-1.0),
                "status": "CONFIRMED" | "REFUTED" | "MIXED" | "EMPTY",
                "reasoning": str,
                "is_jackpot": bool
            }
        """
        if not user_claim or not candidates:
            return self._empty_batch_result("Missing claim or candidates")

        prompt = self._build_batch_prompt(user_claim, candidates)

        try:
            result = await self.llm_client.call_json(
                model=ModelID.NANO,
                input=prompt,
                instructions=self._get_batch_instructions(),
                reasoning_effort="low",
                timeout=20.0,  # Slightly longer for batch
                trace_kind="oracle_batch_validation"
            )

            best_index = int(result.get("best_index", -1))
            relevance_score = float(result.get("relevance_score", 0.0))
            relevance_score = max(0.0, min(1.0, relevance_score))

            # Validate index
            if best_index < 0 or best_index >= len(candidates):
                # No good match found
                return self._empty_batch_result("No suitable match")

            status = self._map_status(result.get("status", ""))
            is_jackpot = relevance_score > JACKPOT_THRESHOLD

            # Extract LLM-determined scores (no more heuristics!)
            verified_score = float(result.get("verified_score", -1.0))
            danger_score = float(result.get("danger_score", -1.0))

            # Warning only - no fallback! If -1, it's a bug and should be visible.
            if verified_score < 0 or danger_score < 0:
                logger.warning("[OracleBatch] ⚠️ LLM did not return verified_score/danger_score. BUG!")

            validated_result = {
                "best_index": best_index,
                "relevance_score": relevance_score,
                "status": status,
                "verified_score": verified_score,
                "danger_score": danger_score,
                "reasoning": result.get("reasoning", ""),
                "is_jackpot": is_jackpot
            }

            Trace.event("oracle.batch_validation", {
                "user_claim": user_claim[:80],
                "num_candidates": len(candidates),
                "best_index": best_index,
                "relevance_score": relevance_score,
                "verified_score": verified_score,
                "danger_score": danger_score,
                "status": status,
                "is_jackpot": is_jackpot
            })

            log_level = logging.INFO if is_jackpot else logging.DEBUG
            logger.log(log_level,
                "[OracleBatch] %d candidates -> best=%d, score=%.2f, v=%.2f, status=%s, jackpot=%s",
                len(candidates), best_index, relevance_score, verified_score, status, is_jackpot
            )

            return validated_result

        except Exception as e:
            logger.warning("[OracleBatch] LLM call failed: %s. Returning MISS.", e)
            return self._empty_batch_result(f"LLM error: {e}")

    def _build_batch_prompt(self, user_claim: str, candidates: list[dict]) -> str:
        """Build batch validation prompt with all candidates."""
        # Format candidates for prompt
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"""
[{i}] Claim: "{c.get('claim_text', '')[:300]}"
    Rating: {c.get('rating', 'Unknown')}
    Publisher: {c.get('publisher', 'Unknown')}
    Summary: {c.get('title', '')[:200]}
"""

        return f"""Analyze these fact-check candidates against the User's Claim.

## User's Claim
"{user_claim[:1500]}"

## Fact-Check Candidates from Database
{candidates_text}

## Task
Find the BEST match (if any) that addresses the User's Claim.

Score the BEST candidate's relevance from 0.0 to 1.0:
- 0.9-1.0: EXACT same claim → JACKPOT (stop search)
- 0.5-0.8: Related claim, useful context → EVIDENCE (add to pack)
- 0.0-0.4: Not relevant, keyword coincidence → MISS (ignore)

If NO candidate scores above 0.5, return best_index: -1

## Critical Checks:
1. Same EVENT and PEOPLE?
2. Same TIMEFRAME?
3. Same core ASSERTION?
4. CONCLUSIVE verdict?

## Determine Status AND Scores (for best match):
- CONFIRMED: Claim is TRUE/VERIFIED → verified_score: 0.85-0.95
- REFUTED: Claim is FALSE/FAKE/DEBUNKED → verified_score: 0.05-0.15
- MIXED: Partially true or NEEDS CONTEXT → verified_score: 0.50-0.70

Determined danger_score (0.0-1.0):
- How harmful/dangerous is this claim if believed?
- High (0.8-1.0): Medical misinfo, scams, hate speech.
- Low (0.0-0.3): Scientific nuances, minor inaccuracies, satire.

IMPORTANT: Rating nuances matter!
- "Pants on Fire" / "Fake" → verified_score: 0.05
- "Mostly False" → verified_score: 0.20
- "Half True" / "Mixture" → verified_score: 0.55
- "Mostly True" → verified_score: 0.75
- "True" → verified_score: 0.90

Output JSON:
{{
    "best_index": 0-{len(candidates)-1} or -1 if no match,
    "relevance_score": 0.0-1.0,
    "status": "CONFIRMED" | "REFUTED" | "MIXED",
    "verified_score": 0.0-1.0,
    "danger_score": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

    def _get_batch_instructions(self) -> str:
        return """You are a strict fact-check relevance analyzer.
Your job is to find the BEST matching fact-check from candidates, if any.
Be CONSERVATIVE: only return high scores (0.9+) for essentially identical claims.
If the best match is still weak (<0.5), return best_index: -1.
Compare candidates AGAINST EACH OTHER to pick the most relevant one."""

    def _empty_batch_result(self, reason: str) -> dict:
        """Return a MISS result for batch validation."""
        return {
            "best_index": -1,
            "relevance_score": 0.0,
            "status": "EMPTY",
            "reasoning": reason,
            "is_jackpot": False
        }

    def _build_prompt(
        self, 
        user_claim: str, 
        oracle_claim: str, 
        oracle_rating: str,
        oracle_summary: str
    ) -> str:
        """Build the validation prompt with strict semantic matching requirements."""
        summary_section = f'\nFact-Check Summary: "{oracle_summary[:500]}"' if oracle_summary else ""

        return f"""Compare the User's Claim with the Fact-Check Result.

## User's Claim
"{user_claim[:1500]}"

## Fact-Check from Database
Claim Reviewed: "{oracle_claim[:1000]}"
Rating: "{oracle_rating}"{summary_section}

## Task
Determine how well the Fact-Check addresses the User's Claim.

Score the relevance from 0.0 to 1.0:
- 1.0: EXACT same claim (same event, same people, same timeframe, same core assertion)
- 0.8-0.9: Same topic and very similar claim, minor differences in wording
- 0.5-0.7: Related topic, but different specific claim (e.g., old hoax vs new claim about same topic)
- 0.2-0.4: Loosely related, shares some keywords but different stories
- 0.0-0.1: Unrelated, keyword coincidence only

## Critical Checks (Lower score if ANY fail):
1. Same EVENT? (e.g., both about Zelenskyy yacht purchase, not just "Zelenskyy")
2. Same TIMEFRAME? (e.g., both about 2024 claim, not one from 2019)
3. Same ASSERTION? (e.g., both claiming he bought it, not one about a different purchase)
4. CONCLUSIVE verdict? (e.g., "False" is conclusive; "Needs context" is not)

## Determine Status AND Scores:
- CONFIRMED: Claim is TRUE/VERIFIED → verified_score: 0.85-0.95
- REFUTED: Claim is FALSE/FAKE/DEBUNKED → verified_score: 0.05-0.15
- MIXED: Partially true or NEEDS CONTEXT → verified_score: 0.50-0.70

Determined danger_score (0.0-1.0):
- How harmful/dangerous is this claim if believed?
- High (0.8-1.0): Medical misinfo, scams, hate speech.
- Low (0.0-0.3): Scientific nuances, minor inaccuracies, satire.

IMPORTANT: Rating nuances matter!
- "Pants on Fire" / "Fake" → verified_score: 0.05
- "Mostly False" → verified_score: 0.20
- "Half True" / "Mixture" → verified_score: 0.55
- "Mostly True" → verified_score: 0.75
- "True" → verified_score: 0.90

Output JSON:
{{
    "relevance_score": 0.0-1.0,
    "status": "CONFIRMED" | "REFUTED" | "MIXED",
    "verified_score": 0.0-1.0,
    "danger_score": 0.0-1.0,
    "reasoning": "Brief explanation of why this score"
}}"""

    def _get_instructions(self) -> str:
        return """You are a strict relevance checker for fact-check results.
Your job is to prevent false positive matches where keywords overlap but the actual claims differ.
Be CONSERVATIVE with high scores (0.8+). Only give 0.9+ if the claims are essentially identical.
A score of 0.5-0.7 means "related but not the same claim" - this is useful context but not conclusive."""

    def _map_status(self, llm_status: str) -> OracleStatus:
        """Map LLM status to OracleStatus. No heuristics - LLM must provide valid status."""
        status_upper = (llm_status or "").upper()

        if status_upper in ("CONFIRMED", "REFUTED", "MIXED"):
            return status_upper  # type: ignore

        # No fallback! If LLM didn't return valid status, it's a bug.
        logger.warning("[OracleValidation] ⚠️ LLM returned invalid status '%s'. BUG!", llm_status)
        return "MIXED"  # Return MIXED but log the bug

    def _empty_result(self, reason: str) -> dict:
        """Return a MISS result for empty/error cases."""
        return {
            "relevance_score": 0.0,
            "status": "EMPTY",
            "reasoning": reason,
            "is_jackpot": False
        }
