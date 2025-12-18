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
    ) -> tuple[list[Claim], bool]:
        """
        Extract atomic verifiable claims from article text.
        """
        text = (text or "").strip()
        if not text:
            return [], False

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
6. Generate EXACTLY 3 search queries for each claim:
   - Query 1 (Specific/Quote): ONLY generate if the claim contains a specific verifiable quote. Format: "exact quote fragment" entity keyword. If no quote exists, fallback to Query 2 style.
   - Query 2 (Event-Based): The MAIN search query. STRICTLY follow the format: Subject Action Object Date/Context. Do NOT use quotes. Do NOT use meta-phrases like "full and final" or "insufficient evidence". Example: Trump orders blockade sanctioned oil tankers Venezuela December 2025.
   - Query 3 (Local): Search in {lang_name} ({lang}) keywords for local coverage.
7. For EACH claim, set "check_oracle": true IF AND ONLY IF the claim discusses rumors, hoaxes, debunking, conspiracy theories, or popular viral myths (e.g. "aliens", "flat earth", "fact check", "fake"). Otherwise false.

Output valid JSON key "claims":
{{
  "claims": [
    {{
      "text": "claim text",
      "type": "core",
      "importance": 0.9,
      "search_queries": ["parametric query", "official query", "local query"],
      "evidence_req": {{
        "needs_primary": true,
        "needs_2_independent": true
      }},
      "check_oracle": false
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
            cache_key = f"claim_extract_v4_{lang}"
            
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
                    check_oracle=bool(rc.get("check_oracle", False)),
                )
                claims.append(c)
                
            # M60 Oracle Optimization: Check if ANY claim needs oracle
            check_oracle = any(c["check_oracle"] for c in claims)
            return claims, check_oracle
            
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
            ], False
