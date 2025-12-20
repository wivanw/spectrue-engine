from spectrue_core.verification.evidence_pack import Claim, EvidenceRequirement, ArticleIntent
from .base_skill import BaseSkill, logger
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX
from spectrue_core.constants import SUPPORTED_LANGUAGES

# M70: Import schema module for structured claims
from spectrue_core.schema import (
    ClaimUnit,
    Assertion,
    Dimension,
    ClaimType,
    ClaimDomain,
    EvidenceRequirementSpec,
    EventQualifiers,
    LocationQualifier,
)

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

# M70: Claim domain mapping
DOMAIN_MAPPING = {
    "Politics": ClaimDomain.POLITICS,
    "Economy": ClaimDomain.FINANCE,
    "War": ClaimDomain.NEWS,
    "Science": ClaimDomain.SCIENCE,
    "Technology": ClaimDomain.TECHNOLOGY,
    "Health": ClaimDomain.HEALTH,
    "Environment": ClaimDomain.SCIENCE,
    "Society": ClaimDomain.NEWS,
    "Sports": ClaimDomain.SPORTS,
    "Culture": ClaimDomain.ENTERTAINMENT,
    "Other": ClaimDomain.OTHER,
}

# M70: Claim type mapping from legacy
CLAIM_TYPE_MAPPING = {
    "core": ClaimType.EVENT,
    "numeric": ClaimType.NUMERIC,
    "timeline": ClaimType.TIMELINE,
    "attribution": ClaimType.ATTRIBUTION,
    "sidefact": ClaimType.OTHER,
}


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

    # ─────────────────────────────────────────────────────────────────────────
    # M70: Schema-First Extraction (Spec Producer)
    # ─────────────────────────────────────────────────────────────────────────

    async def extract_claims_structured(
        self,
        text: str,
        *,
        lang: str = "en",
        max_claims: int = 5,
    ) -> tuple[list[ClaimUnit], bool, ArticleIntent]:
        """
        M70: Schema-constrained claim extraction.
        
        This is the SPEC PRODUCER. LLM acts as parser + interpreter:
        - Fills structured ClaimUnit fields
        - Classifies each field as FACT / CONTEXT / INTERPRETATION
        - Does NOT decide truth (that's for scoring)
        
        Key Design:
        - time_reference ≠ location (the core bug fix)
        - Each assertion has explicit dimension
        - Schema is stable contract for downstream consumers
        
        Returns:
            tuple of (claim_units, should_check_oracle, article_intent)
        """
        text = (text or "").strip()
        if not text:
            return [], False, "news"

        text_excerpt = text[:8000] if len(text) > 8000 else text
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        topics_str = ", ".join(TOPIC_GROUPS)

        # M70: Schema-Constrained Generation Prompt
        instructions = f"""You are a SCHEMA PARSER for fact-checking.

## YOUR ROLE
You are a **parser + interpreter**, NOT a judge.
- Fill structured fields from text
- Classify each field as FACT / CONTEXT / INTERPRETATION
- Do NOT decide if claims are true or false

## CRITICAL: FACT vs CONTEXT DISTINCTION

### FACT (dimension: "FACT")
- Claims about the world that MUST be verified by evidence
- Examples: "fight in Miami", "price is $100", "John said X"
- Verification: STRICT (can be verified/refuted)

### CONTEXT (dimension: "CONTEXT")  
- Contextual framing that provides background
- Examples: "Ukraine time", "Kyiv timezone", "for European viewers"
- Verification: SOFT (informational only, rarely refuted)

### THE BUG TO AVOID
❌ WRONG: "(в Україні)" after time → location.country = "Ukraine"
✅ CORRECT: "(в Україні)" after "03:00 (за Києвом)" → time_reference = "Ukraine time" (CONTEXT)

Time zone references are NOT location claims!

## OUTPUT SCHEMA

```json
{{
  "article_intent": "news|evergreen|official|opinion|prediction",
  "claims": [
    {{
      "id": "c1",
      "domain": "sports|science|politics|health|finance|news|other",
      "claim_type": "event|attribution|numeric|definition|timeline|other",
      
      "subject": "Anthony Joshua",
      "predicate": "scheduled_fight_against", 
      "object": "Jake Paul",
      
      "assertions": [
        {{
          "key": "event.time",
          "value": "03:00",
          "value_raw": "03:00 (за Києвом)",
          "dimension": "FACT"
        }},
        {{
          "key": "event.time_reference",
          "value": "Kyiv time",
          "value_raw": "(за Києвом)",
          "dimension": "CONTEXT"
        }},
        {{
          "key": "event.location.city",
          "value": "Miami",
          "value_raw": "in Miami",
          "dimension": "FACT"
        }}
      ],
      
      "qualifiers": {{
        "event_date": null,
        "event_time": null,
        "time_reference": "Kyiv time",
        "location": {{
          "city": "Miami",
          "country": "USA",
          "is_inferred": false
        }},
        "participants": ["Anthony Joshua", "Jake Paul"]
      }},
      
      "importance": 0.9,
      "check_worthiness": 0.9,
      "extraction_confidence": 0.95,
      
      "text": "original text excerpt",
      "normalized_text": "Self-sufficient version with context",
      "topic_group": "{topics_str}",
      "topic_key": "Joshua vs Paul Fight",
      
      "query_candidates": [
        {{"text": "Joshua Paul fight Miami official announcement", "role": "CORE", "score": 1.0}},
        {{"text": "Joshua Paul boxing match date location", "role": "NUMERIC", "score": 0.8}}
      ],
      "search_method": "news",
      "check_oracle": false
    }}
  ]
}}
```

## ASSERTION KEY CONVENTIONS
- `event.time` - When it happens (FACT)
- `event.time_reference` - Timezone context (CONTEXT)
- `event.location.city` - Where it happens (FACT)
- `event.location.venue` - Specific venue (FACT)
- `event.participants` - Who is involved (FACT)
- `attribution.quote` - What was said (FACT)
- `attribution.speaker` - Who said it (FACT)
- `numeric.value` - The number (FACT)
- `numeric.unit` - Unit of measurement (CONTEXT)

## RULES

1. **Extract only explicit claims** - don't infer unstated facts
2. **Mark inferred fields** - if you infer something, set `is_inferred: true`
3. **Preserve raw text** - store original excerpt in `value_raw`
4. **Simple facts are valid** - single FACT assertion is fine for simple claims
5. **Language**: Generate in {lang_name} for text/normalized_text, English for queries

You MUST respond in valid JSON.
"""

        prompt = f"""Parse this article into structured claims.
For each claim, identify FACT assertions (verifiable) and CONTEXT assertions (informational).

CRITICAL: Time zone references like "(в Україні)", "(за Києвом)" are CONTEXT, not location!

ARTICLE:
{text_excerpt}

Return structured ClaimUnits in JSON format.
"""

        try:
            cache_key = f"claim_schema_v1_{lang}"

            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="medium",  # Higher effort for schema parsing
                cache_key=cache_key,
                timeout=60.0,  # Longer timeout for complex parsing
                trace_kind="claim_extraction_structured",
            )

            raw_claims = result.get("claims", [])
            claim_units: list[ClaimUnit] = []

            # M63: Extract article intent
            raw_intent = result.get("article_intent", "news")
            if raw_intent not in ARTICLE_INTENTS:
                raw_intent = "news"
            article_intent: ArticleIntent = raw_intent  # type: ignore

            for idx, rc in enumerate(raw_claims):
                if not isinstance(rc, dict):
                    continue

                claim_id = rc.get("id") or f"c{idx + 1}"

                # Parse assertions
                assertions: list[Assertion] = []
                for ra in rc.get("assertions", []):
                    if not isinstance(ra, dict):
                        continue
                    
                    # Parse dimension (LLM decides this, we just validate)
                    dim_str = ra.get("dimension", "FACT").upper()
                    if dim_str == "CONTEXT":
                        dimension = Dimension.CONTEXT
                    elif dim_str == "INTERPRETATION":
                        dimension = Dimension.INTERPRETATION
                    else:
                        dimension = Dimension.FACT

                    assertions.append(Assertion(
                        key=ra.get("key", "claim.text"),
                        value=ra.get("value"),
                        value_raw=ra.get("value_raw"),
                        dimension=dimension,
                        importance=float(ra.get("importance", 1.0)),
                        is_inferred=bool(ra.get("is_inferred", False)),
                        evidence_requirement=EvidenceRequirementSpec(
                            needs_primary=bool(ra.get("needs_primary", False)),
                            needs_2_independent=bool(ra.get("needs_2_independent", False)),
                        ),
                    ))

                # Parse qualifiers
                qualifiers = None
                raw_qual = rc.get("qualifiers")
                if isinstance(raw_qual, dict):
                    location = None
                    raw_loc = raw_qual.get("location")
                    if isinstance(raw_loc, dict):
                        location = LocationQualifier(
                            venue=raw_loc.get("venue"),
                            city=raw_loc.get("city"),
                            region=raw_loc.get("region"),
                            country=raw_loc.get("country"),
                            is_inferred=bool(raw_loc.get("is_inferred", False)),
                        )

                    qualifiers = EventQualifiers(
                        time_reference=raw_qual.get("time_reference"),
                        location=location,
                        participants=raw_qual.get("participants", []),
                    )

                # Map claim type
                raw_type = rc.get("claim_type", "other").lower()
                claim_type = CLAIM_TYPE_MAPPING.get(raw_type, ClaimType.OTHER)
                # Also try direct mapping for M70 types
                for ct in ClaimType:
                    if ct.value == raw_type:
                        claim_type = ct
                        break

                # Map domain
                topic_group = rc.get("topic_group", "Other")
                if topic_group not in TOPIC_GROUPS:
                    topic_group = "Other"
                domain = DOMAIN_MAPPING.get(topic_group, ClaimDomain.OTHER)
                # Also try direct mapping
                raw_domain = rc.get("domain", "").lower()
                for cd in ClaimDomain:
                    if cd.value == raw_domain:
                        domain = cd
                        break

                # Create ClaimUnit
                claim_unit = ClaimUnit(
                    id=claim_id,
                    domain=domain,
                    claim_type=claim_type,
                    subject=rc.get("subject"),
                    predicate=rc.get("predicate", ""),
                    object=rc.get("object"),
                    qualifiers=qualifiers,
                    assertions=assertions,
                    importance=float(rc.get("importance", 0.5)),
                    check_worthiness=float(rc.get("check_worthiness", 0.5)),
                    extraction_confidence=float(rc.get("extraction_confidence", 1.0)),
                    text=rc.get("text", ""),
                    normalized_text=rc.get("normalized_text", rc.get("text", "")),
                    topic_group=topic_group,
                    topic_key=rc.get("topic_key", topic_group),
                    language=lang,
                )

                claim_units.append(claim_unit)

                # Debug logging
                fact_count = len(claim_unit.get_fact_assertions())
                context_count = len(claim_unit.get_context_assertions())
                logger.debug(
                    "[M70] Claim %s: %d FACT, %d CONTEXT assertions | type=%s",
                    claim_id, fact_count, context_count, claim_type.value
                )

            # Log summary
            total_facts = sum(len(c.get_fact_assertions()) for c in claim_units)
            total_context = sum(len(c.get_context_assertions()) for c in claim_units)
            logger.info(
                "[M70] Extracted %d claims: %d FACT assertions, %d CONTEXT assertions",
                len(claim_units), total_facts, total_context
            )

            # Check oracle from query_candidates
            check_oracle = any(
                bool(rc.get("check_oracle", False))
                for rc in raw_claims
                if isinstance(rc, dict)
            )

            return claim_units, check_oracle, article_intent

        except Exception as e:
            logger.warning("[M70] Structured extraction failed: %s. Using fallback.", e)
            # Fallback: Create minimal ClaimUnit
            fallback_text = text[:300] + "..." if len(text) > 300 else text
            return [
                ClaimUnit(
                    id="c1",
                    claim_type=ClaimType.OTHER,
                    text=fallback_text,
                    normalized_text=fallback_text,
                    topic_group="Other",
                    topic_key="General",
                    importance=1.0,
                    check_worthiness=0.5,
                    assertions=[
                        Assertion(
                            key="claim.text",
                            value=fallback_text,
                            dimension=Dimension.FACT,
                        )
                    ],
                    language=lang,
                )
            ], False, "news"

