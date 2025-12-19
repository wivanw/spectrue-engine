from .base_skill import BaseSkill, logger
from spectrue_core.verification.evidence_pack import Claim, SearchResult

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
        
        prompt = f"""You are a semantic relevance filter for a fact-checking system.
        
Input Claims:
{claims_text}

Search Results Snippets:
{snippets_text}

Task: Determine if the search snippets contain RELEVANT information to verify or debunk these specific claims.
- Ignore minor keyword matches if the context is unrelated.
- Look for semantic relevance (facts, data, statements related to the claim event/topic).
- "Relevant" means the text confirms, denies, or provides specific context about the events in the claim.

Output JSON:
{{
  "is_relevant": true/false,
  "reason": "Short explanation why relevant or not"
}}
"""
        
        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a relevance filter.",
                reasoning_effort="low",
                timeout=15.0,
                trace_kind="semantic_gating"
            )
            
            is_relevant = bool(result.get("is_relevant", False))
            reason = result.get("reason", "No reason provided")
            
            if not is_relevant:
                logger.info("[M66] Semantic Gating: Results rejected. Reason: %s", reason)
            else:
                logger.debug("[M66] Semantic Gating: Results accepted. Reason: %s", reason)
                
            return {
                "is_relevant": is_relevant,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning("[M66] Semantic Gating failed: %s. Assuming relevant.", e)
            return {"is_relevant": True, "reason": "Error in check, fail open"}
