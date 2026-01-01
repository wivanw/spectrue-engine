# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from .base_skill import BaseSkill, logger
from spectrue_core.verification.evidence_pack import Claim

class RelevanceSkill(BaseSkill):
    """
    Semantic Gating for Search Results.
    Filters out irrelevant search results using lightweight LLM checks before expensive processing.
    """

    async def verify_search_relevance(
        self, 
        claims: list[Claim], 
        search_results: list[dict]
    ) -> dict:
        """
        Check if search results contain relevant information to verify the claims.
        
        Args:
            claims: List of extracted claims
            search_results: List of SearchResult dicts
            
        Returns:
            Dict: {"is_relevant": bool, "reason": str}
        """
        if not claims or not search_results:
            return {"is_relevant": False, "reason": "No input data"}

        # Prepare context for LLM
        # Use top 3 claims
        claims_text = "\n".join([
            f"- {c.get('normalized_text') or c.get('text', '')}"
            for c in claims[:3]
        ])

        # Use top 3 search snippets
        snippets_text = "\n".join([
            f"Source: {res.get('title', 'Unknown')}\nSnippet: {res.get('snippet', '')}\n---" 
            for res in search_results[:3]
        ])

        prompt = f"""You are a semantic relevance router for a fact-checking system.
        
Input Claims:
{claims_text}

Search Results Snippets:
{snippets_text}

Task: Determine the semantic relationship between the snippets and the claims.
Classify the MATCH TYPE:
- "EXACT": The snippets explicitly mention the specific event, stats, or quotes in the claims. High verification value.
- "TOPIC": The snippets discuss the same entities/topic but do NOT confirm/deny the specific details. Useful for context.
- "UNRELATED": Completely unrelated (different entities, different context, spam).

Return JSON:
{{
  "match_type": "EXACT" | "TOPIC" | "UNRELATED",
  "reason": "Short explanation"
}}
"""

        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a semantic router. Be generous with TOPIC matches for context, strict with EXACT matches.",
                reasoning_effort="low",
                timeout=15.0,
                trace_kind="semantic_gating"
            )

            match_type = result.get("match_type", "TOPIC")
            reason = result.get("reason", "No reason provided")

            # Two-Level Gating
            # Accept both EXACT and TOPIC as "RELEVANT" to prevent coverage loss
            is_relevant = match_type in ("EXACT", "TOPIC")
            status = "RELEVANT" if is_relevant else "OFF_TOPIC"

            logger.debug("[M74] Semantic Router: match_type=%s → Status=%s. Reason: %s", match_type, status, reason)

            return {
                "status": status,
                "match_type": match_type,
                "is_relevant": is_relevant,
                "reason": reason
            }

        except Exception as e:
            logger.warning("[M66] Semantic Gating failed: %s. Assuming relevant.", e)
            return {"status": "RELEVANT", "is_relevant": True, "reason": "Error in check, fail open"}
