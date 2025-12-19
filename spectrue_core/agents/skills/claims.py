from spectrue_core.verification.evidence_pack import Claim, EvidenceRequirement, ArticleIntent
from .base_skill import BaseSkill, logger
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX
from spectrue_core.constants import SUPPORTED_LANGUAGES

# M62: Available topic groups for claim classification
TOPIC_GROUPS = [
    "Politics", "Economy", "War", "Science", "Technology", 
    "Health", "Environment", "Society", "Sports", "Culture", "Other"
]

# M62+: Search intent types for strategist approach
SEARCH_INTENTS = [
    "scientific_fact",      # Peer-reviewed research, studies
    "official_statement",   # Government, company announcements
    "breaking_news",        # Recent events, developing stories
    "historical_event",     # Past events, dates, timelines
    "quote_attribution",    # Who said what
    "prediction_opinion",   # Forecasts, betting, expectations
    "viral_rumor",          # Hoaxes, myths, debunking needed
]

# M63: Article intent types for Oracle triggering
# - CHECK Oracle: news, evergreen, official
# - SKIP Oracle: opinion, prediction
ARTICLE_INTENTS = ["news", "evergreen", "official", "opinion", "prediction"]


class ClaimExtractionSkill(BaseSkill):
    
    async def extract_claims(
        self,
        text: str,
        *,
        lang: str = "en",
        max_claims: int = 5,
    ) -> tuple[list[Claim], bool, ArticleIntent]:
        """
        Extract atomic verifiable claims from article text.
        
        M62: Context-aware claims with normalized_text, topic_group, check_worthiness
        M63: Returns article_intent for Oracle triggering
        M64: Topic-Aware Round-Robin support (topic_key, query_candidates)
             + Negative Constraints (Gambling Guardrail via LLM)
        
        Returns:
            tuple of (claims, should_check_oracle, article_intent)
        """
        text = (text or "").strip()
        if not text:
            return [], False, "news"  # Default intent

        # Limit input to prevent token overflow
        text_excerpt = text[:8000] if len(text) > 8000 else text
        
        # M57: Resolve language name for bilingual query generation
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        
        topics_str = ", ".join(TOPIC_GROUPS)
        intents_str = ", ".join(SEARCH_INTENTS)
        
        # Updated Strategist Prompt with Negative Constraints & Query Candidates
        instructions = f"""You are an expert Fact-Checking Search Strategist.
Your goal is to extract verifiable claims AND develop optimal SEARCH STRATEGIES for each.

## NEGATIVE CONSTRAINTS (CRITICAL):
1. Do NOT generate queries seeking betting odds, gambling coefficients, or bookmaker predictions.
2. If the user asks for a prediction (e.g. sports), search for "expert analysis" or "official announcement", NOT "betting site".
3. EXCEPTION: If the user explicitly asks about "gambling regulations" or "corruption in betting", then betting terms ARE allowed.

## STEP 1: EXTRACT CLAIMS
For each claim in the article:
1. Extract the core factual assertion (not opinions)
2. Preserve EXACT numbers, dates, names, locations
3. Classify type: "core", "numeric", "timeline", "attribution", "sidefact"
   - "sidefact" = background info that will be SKIPPED from search

## STEP 2: THINK LIKE A SEARCH STRATEGIST (Chain of Thought)
For each claim, REASON about:

1. **Intent Analysis**: What type of claim is this?
   - {intents_str}

2. **Primary Authority**: Where would the ORIGINAL evidence exist?
   - Science/Medicine → Journals. Best in ENGLISH.
   - Local Politics → Local news. Best in LOCAL language.
   - Sports → Official sites, Interviews. AVOID betting sites!

3. **Language Strategy**: 
   - Scientific facts → Search in ENGLISH
   - Local news → Search in LOCAL language ({lang_name})

4. **Search Method Selection**:
   - **"news"**: Recent events (last 30 days), politics, unfolding situations.
   - **"general_search"**: Scientific facts, historical data, definitions, established context.
   - **"academic"**: specialized studies (if required).

## STEP 3: GENERATE QUERY CANDIDATES
Generate 2-3 query candidates for each claim with SPECIFIC ROLES:

1. **CORE** (Required): "Event + Date + Action" (Priority 1.0)
   Example: "Hubble Fomalhaut collision December 2024"
   
2. **NUMERIC** (If numbers exist): "Metric + Value + 'Official Data'" (Priority 0.8)
   Example: "Bitcoin price $42000 official exchange rate"
   
3. **ATTRIBUTION** (If quotes exist): "Person + 'Quote' + Source" (Priority 0.7)
   Example: "NASA administrator statement Fomalhaut discovery"

4. **LOCAL** (Optional): Best local language query (Priority 0.5)

Also assign for each claim:
- **topic_group**: High-level category ({topics_str})
- **topic_key**: Short, consistent subject tag (e.g., "Fomalhaut System", "Bitcoin Price") - used for round-robin coverage.

## STEP 4: CLASSIFY ARTICLE INTENT
Determine the OVERALL article intent for Oracle triggering:
- "news": Current events, breaking news → CHECK Oracle
- "evergreen": Science facts, historical claims → CHECK Oracle  
- "official": Government notifications → CHECK Oracle
- "opinion": Editorial, commentary → SKIP Oracle
- "prediction": Future events, forecasts → SKIP Oracle

## OUTPUT FORMAT

```json
{{
  "article_intent": "news",
  "claims": [
    {{
      "text": "Original text from article",
      "normalized_text": "Self-sufficient: Trump announced tariffs on China Dec 19",
      "type": "core",
      "topic_group": "Politics",
      "topic_key": "Trump Tariffs",
      "importance": 0.9,
      "check_worthiness": 0.9,
      "search_strategy": {{
        "intent": "official_statement",
        "reasoning": "Needs official confirmation or major news",
        "best_language": "en"
      }},
      "query_candidates": [
        {{"text": "Trump China tariffs announcement 2025", "role": "CORE", "score": 1.0}},
        {{"text": "Трамп мита Китай 2025", "role": "LOCAL", "score": 0.5}}
      ],
      "search_method": "news",
      "search_queries": [
        "Trump China tariffs announcement 2025",
        "Трамп мита Китай 2025"
      ],
      "evidence_req": {{
        "needs_primary": true,
        "needs_2_independent": true
      }},
      "check_oracle": false
    }}
  ]
}}
```

## TYPE CLASSIFICATION
- "core": Main controversial claim (importance 0.9-1.0)
- "numeric": Contains specific numbers/stats (importance 0.7-0.8)
- "timeline": Dates, sequences (importance 0.7)
- "attribution": Quote - WHO said WHAT (importance 0.6)
- "sidefact": Background, common knowledge (SKIP search, importance 0.3)

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""
        prompt = f"""Extract 3-{max_claims} atomic verifiable claims from this article.
Apply Topic-Aware logic: group by topic_key and generate query_candidates with roles.

ARTICLE:
{text_excerpt}

Return the result in JSON format.
"""
        try:
            # Updated cache key version for new prompt structure
            cache_key = f"claim_strategist_v2_{lang}"

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
            
            # M63: Extract article intent
            raw_intent = result.get("article_intent", "news")
            if raw_intent not in ARTICLE_INTENTS:
                raw_intent = "news"  # Default to news (check Oracle)
            article_intent: ArticleIntent = raw_intent  # type: ignore
            
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
                
                # M62: Extract new fields with safe defaults
                normalized = rc.get("normalized_text", "") or rc.get("text", "")
                topic = rc.get("topic_group", "Other") or "Other"
                # Validate topic_group against allowed list
                if topic not in TOPIC_GROUPS:
                    topic = "Other"
                
                # M64: topic_key extraction
                topic_key = rc.get("topic_key") or topic  # Fallback to group if key missing
                
                worthiness = rc.get("check_worthiness")
                if worthiness is None:
                    # Fallback: derive from importance
                    worthiness = float(rc.get("importance", 0.5))
                else:
                    worthiness = float(worthiness)
                worthiness = max(0.0, min(1.0, worthiness))  # Clamp to 0-1
                
                # M62+: Extract search strategy if present
                strategy = rc.get("search_strategy", {})
                
                c = Claim(
                    id=f"c{idx+1}",
                    text=rc.get("text", ""),
                    type=rc.get("type", "core"),  # type: ignore
                    importance=float(rc.get("importance", 0.5)),
                    evidence_requirement=req,
                    search_queries=rc.get("search_queries", []),
                    check_oracle=bool(rc.get("check_oracle", False)),
                    # M62: New fields
                    normalized_text=normalized,
                    topic_group=topic,
                    check_worthiness=worthiness,
                    # M64: New fields
                    topic_key=topic_key,
                    query_candidates=rc.get("query_candidates", []),
                    # M66: Smart Routing
                    search_method=rc.get("search_method", "general_search")
                )
                
                # Log strategy for debugging
                if strategy:
                    intent = strategy.get("intent", "?")
                    reasoning = strategy.get("reasoning", "")[:50]
                    logger.debug(
                        "[Strategist] Claim %d: intent=%s, topic_key=%s | %s",
                        idx+1, intent, topic_key, reasoning
                    )
                
                claims.append(c)
            
            # Log topic and strategy distribution
            topics_found = [c.get("topic_key", "?") for c in claims]
            logger.info("[Claims] Extracted %d claims. Topics keys: %s", len(claims), topics_found)
                
            # M60 Oracle Optimization: Check if ANY claim needs oracle
            check_oracle = any(c.get("check_oracle", False) for c in claims)
            
            # M63: Log intent for debugging
            logger.info("[Claims] Article intent: %s (check_oracle=%s)", article_intent, check_oracle)
            
            return claims, check_oracle, article_intent
            
        except Exception as e:
            logger.warning("[M48] Claim extraction failed: %s. Using fallback.", e)
            # Fallback: Treat entire text as one core claim
            fallback_text = text[:300] + "..." if len(text) > 300 else text
            return [
                Claim(
                    id="c1",
                    text=fallback_text,
                    normalized_text=fallback_text,
                    type="core",
                    topic_group="Other",
                    topic_key="General",
                    importance=1.0,
                    check_worthiness=0.5,
                    evidence_requirement=EvidenceRequirement(
                        needs_primary_source=False,
                        needs_independent_2x=True
                    ),
                    search_queries=[],
                    query_candidates=[],
                )
            ], False, "news"  # Default intent on fallback
