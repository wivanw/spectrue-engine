from spectrue_core.verification.evidence_pack import Claim, EvidenceRequirement
from .base_skill import BaseSkill, logger
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX
from spectrue_core.constants import SUPPORTED_LANGUAGES

class ClaimExtractionSkill(BaseSkill):
    
    async def extract_claims(
        self,
        text: str,
        *,
        lang: str = "en",
        max_claims: int = 5, # M56: Reduced from 7 to 5 for speed
    ) -> list[Claim]:
        """
        Extract atomic verifiable claims from article text.
        """
        text = (text or "").strip()
        if not text:
            return []

        # Limit input to prevent token overflow
        text_excerpt = text[:8000] if len(text) > 8000 else text
        
        # M57: Resolve language name for bilingual query generation
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")

        # Move static rules to instructions for prefix caching (M56)
        # We append the UNIVERSAL_METHODOLOGY_APPENDIX to ensure the prefix is heavy (>1024 tokens) and consistent.
        instructions = f"""You are a claim extraction assistant.
Rules:
1. Each claim must be independently verifiable (a single fact, not an opinion).
2. Use neutral, indicative phrasing (e.g. "Event X date is Y").
3. Preserve EXACT numbers, dates, names.
4. Classify type: "core", "numeric", "timeline", "attribution", "sidefact".
5. Importance: 0.9-1.0 for core, 0.4 for side.
6. Generate EXACTLY 2 search queries for each claim:
   - Query 1: MUST be in ENGLISH (for international sources).
   - Query 2: MUST be in {lang_name} ({lang}) (for local sources).
   - Both queries should be factual and specific.

Output valid JSON key "claims":
{{
  "claims": [
    {{
      "text": "claim text",
      "type": "core",
      "importance": 0.9,
      "search_queries": ["query1", "query2"],
      "evidence_req": {{
        "needs_primary": true,
        "needs_2_independent": true
      }}
    }}
  ]
}}

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""
        prompt = f"""Extract 3-{max_claims} atomic verifiable claims from this article.

ARTICLE:
{text_excerpt}
"""
        try:
            cache_key = f"claim_extract_v3_{lang}"
            
            # Ensure "JSON" appears in input (OpenAI Responses API requirement)
            prompt += "\n\nReturn the result in JSON format."

            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                cache_key=cache_key,
                timeout=45.0,
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
