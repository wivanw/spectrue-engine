from .base_skill import BaseSkill, logger
from spectrue_core.verification.evidence_pack import Claim

class RelevanceSkill(BaseSkill):
    """
    M66: Semantic Gating for Search Results.
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
        claims_text = "\n".join([f"- {c.get('text', '')}" for c in claims[:3]])
        
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
Return one of these statuses:
- "RELEVANT": The snippets discuss the same entities, events, or topic. Useful for verification.
- "OFF_TOPIC": Completely unrelated (e.g., search discussing football, claims discussing physics).
- "TOO_BROAD": Snippets are too generic (e.g., dictionary definitions) when specific news is needed.
- "NOT_FOUND": Snippets explicitly say "No results found" or similar.

Output JSON:
{{
  "status": "RELEVANT" | "OFF_TOPIC" | "TOO_BROAD" | "NOT_FOUND",
  "reason": "Short explanation"
}}
"""
        
        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a semantic router.",
                reasoning_effort="low",
                timeout=15.0,
                trace_kind="semantic_gating"
            )
            
            status = result.get("status", "RELEVANT")
            reason = result.get("reason", "No reason provided")
            
            # Map legacy boolean for backward compat if needed, but primary is status
            is_relevant = status == "RELEVANT"
            
            logger.info("[M67] Semantic Router: Status=%s. Reason: %s", status, reason)
                
            return {
                "status": status,
                "is_relevant": is_relevant, # Keep for compatibility
                "reason": reason
            }
            
        except Exception as e:
            logger.warning("[M66] Semantic Gating failed: %s. Assuming relevant.", e)
            return {"status": "RELEVANT", "is_relevant": True, "reason": "Error in check, fail open"}
