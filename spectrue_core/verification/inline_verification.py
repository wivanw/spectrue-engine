from spectrue_core.verification.evidence_pack import Claim
from spectrue_core.agents.skills.base_skill import BaseSkill, logger

class InlineVerificationSkill(BaseSkill):
    """
    Skill for verifying specific types of evidence inline (e.g., Social Media Identity).
    Part of Tier A' Verification (M67).
    """

    async def verify_social_statement(
        self, 
        claim: Claim, 
        social_snippet: str | None, 
        profile_url: str | None
    ) -> dict:
        """
        Verify if a social media snippet is an official statement from the claim subject.
        
        Performs two independent checks:
        1. Identity Verification: Is this account the official account of the entity?
        2. Content Support: Does the text explicitly support the claim?
        
        Returns:
            {
                "is_identity_verified": float, # 0.0 - 1.0
                "is_statement_supported": float, # 0.0 - 1.0
                "tier": "A'" | "D" 
            }
        """
        if not social_snippet or not hasattr(claim, 'text'):
            return {"is_identity_verified": 0.0, "is_statement_supported": 0.0, "tier": "D"}

        # 1. Identity Check
        # We ask LLM to verify if the URL/Context matches the entity in the claim
        identity_score = await self._check_identity(claim.get("text"), profile_url, social_snippet)
        
        # 2. Content Support Check
        # We ask LLM if the text supports the claim
        support_score = await self._check_support(claim.get("text"), social_snippet)
        
        # 3. Determine Tier
        # Strict Rule: Both must be > 0.8 to qualify for Tier A' (0.75 cap)
        tier = "D"
        if identity_score > 0.8 and support_score > 0.8:
            tier = "A'"
            
        logger.info(
            "[M67] Social Verification: Identity=%.2f, Support=%.2f -> Tier %s",
            identity_score, support_score, tier
        )
            
        return {
            "is_identity_verified": identity_score,
            "is_statement_supported": support_score,
            "tier": tier
        }

    async def _check_identity(self, claim_text: str, url: str | None, snippet: str) -> float:
        prompt = f"""Verify Social Media Identity.

Claim Context: "{claim_text}"
Profile URL: {url or 'Unknown'}
Snippet Context: "{snippet[:500]}"

Task: Determine if this social media account BELONGS to the main entity mentioned in the claim.
- If URL is official (e.g. twitter.com/ElonMusk for Elon Musk): Score 1.0
- If snippet implies official status ("Official account of..."): Score 0.9
- If it looks like a fan page/random user: Score 0.1
- If unsure: Score 0.5

Output JSON: {{ "score": float, "reason": str }}
"""
        return await self._run_check(prompt, "identity_check")

    async def _check_support(self, claim_text: str, snippet: str) -> float:
        prompt = f"""Verify Statement Support.

Claim: "{claim_text}"
Source Text: "{snippet[:1000]}"

Task: Determine if the Source Text EXPLICITLY confirms the Claim.
- Explicit confirmation in first person ("I did X", "We announce Y"): Score 1.0
- Strong support but third person: Score 0.8
- Vague/Unrelated: Score 0.2
- Contradicts: Score 0.0

Output JSON: {{ "score": float, "reason": str }}
"""
        return await self._run_check(prompt, "support_check")

    async def _run_check(self, prompt: str, kind: str) -> float:
        try:
            res = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a strict verification auditor.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind=kind
            )
            return float(res.get("score", 0.0))
        except Exception as e:
            logger.warning("Check %s failed: %s", kind, e)
            return 0.0
