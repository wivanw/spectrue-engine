from spectrue_core.verification.evidence_pack import Claim, EvidenceRequirement
from .base_skill import BaseSkill, logger

class ClaimExtractionSkill(BaseSkill):
    
    async def extract_claims(
        self,
        text: str,
        *,
        lang: str = "en",
        max_claims: int = 7,
    ) -> list[Claim]:
        """
        Extract atomic verifiable claims from article text.
        """
        text = (text or "").strip()
        if not text:
            return []

        # Limit input to prevent token overflow
        text_excerpt = text[:8000] if len(text) > 8000 else text

        prompt = f"""Extract 3-{max_claims} atomic verifiable claims from this article.
Rules:
1. Each claim must be independently verifiable (a single fact, not an opinion).
2. Use neutral, indicative phrasing (e.g. "Event X date is Y", not "Event X will happen"). Avoid future tense predictions.
3. Preserve EXACT numbers, dates, names, and quotes from the original.
4. Classify each claim by type:
   - "core": Main factual assertion of the article
   - "numeric": Claims with specific numbers/statistics (amounts, percentages, distances)
   - "timeline": Claims about dates, deadlines, sequences, or timing
   - "attribution": Claims about who said or did something (quotes, statements)
   - "sidefact": Secondary supporting facts (background info, context)
4. Assign importance (0.0-1.0): core claims = 0.9-1.0, sidefacts = 0.3-0.5
5. For claims requiring strong evidence, set evidence_req fields:
   - needs_primary: true if claim needs official/primary source
   - needs_2_independent: true if claim needs 2+ independent sources
   - needs_quote: true if claim involves a specific quote
   - needs_recent: true if claim is time-sensitive news
6. Generate 2 optimal Google search queries for verifying THIS claim (in the same language).

Output valid JSON with key "claims" (array of objects):
{{
  "claims": [
    {{
      "text": "claim text here",
      "type": "core|numeric|timeline|attribution|sidefact",
      "importance": 0.9,
      "search_queries": ["query 1", "query 2"],
      "evidence_req": {{
        "needs_primary": false,
        "needs_2_independent": true,
        "needs_quote": false,
        "needs_recent": true
      }}
    }}
  ]
}}

ARTICLE:
{text_excerpt}
"""
        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a claim extraction assistant. Extract verifiable claims from articles.",
                reasoning_effort="low",
                cache_key="claim_extract_v1",
                timeout=25.0,
                trace_kind="claim_extraction",
            )
            
            raw_claims = result.get("claims", [])
            claims: list[Claim] = []
            
            for idx, rc in enumerate(raw_claims):
                if not isinstance(rc, dict) or not rc.get("text"):
                    continue
                
                req_raw = rc.get("evidence_req", {})
                req = EvidenceRequirement(
                    needs_primary_source=bool(req_raw.get("needs_primary")),
                    needs_independent_2x=bool(req_raw.get("needs_2_independent")),
                    needs_quote_verification=bool(req_raw.get("needs_quote")),
                    is_time_sensitive=bool(req_raw.get("needs_recent")),
                )
                
                c = Claim(
                    id=f"c{idx+1}",
                    text=rc.get("text", ""),
                    type=rc.get("type", "core"),  # type: ignore
                    importance=float(rc.get("importance", 0.5)),
                    evidence_requirement=req,
                    search_queries=rc.get("search_queries", []),
                )
                claims.append(c)
                
            return claims
            
        except Exception as e:
            logger.warning("[M48] Claim extraction failed: %s. Using fallback.", e)
            # Fallback: Treat entire text as one core claim
            fallback_text = text[:300] + "..." if len(text) > 300 else text
            return [
                Claim(
                    id="c1",
                    text=fallback_text,
                    type="core",
                    importance=1.0,
                    evidence_requirement=EvidenceRequirement(
                        needs_primary_source=False,
                        needs_independent_2x=True
                    ),
                    search_queries=[],
                )
            ]
