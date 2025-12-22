from spectrue_core.verification.evidence_pack import Claim, ClaimAnchor, EvidenceRequirement, ArticleIntent
from .base_skill import BaseSkill, logger
from spectrue_core.utils.text_chunking import TextChunk
from spectrue_core.utils.trace import Trace
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
from spectrue_core.schema.evidence import EvidenceNeedType

# M80: Import claim metadata types
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    ClaimRole,
    VerificationTarget,
    EvidenceChannel,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    default_claim_metadata,
)

from spectrue_core.agents.skills.claim_metadata_parser import (
    parse_claim_metadata as parse_claim_metadata_v1,
    default_channels as default_channels_v1,
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
    
    # M73.5: Dynamic timeout constants
    BASE_TIMEOUT_SEC = 35.0     # Minimum timeout
    TIMEOUT_PER_1K_CHARS = 2.0  # Additional seconds per 1000 chars
    MAX_TIMEOUT_SEC = 75.0      # Maximum timeout cap
    
    def _calculate_timeout(self, text_len: int, *, base_offset: float = 0.0) -> float:
        """
        Calculate dynamic timeout based on text length.
        
        Args:
            text_len: Length of input text in characters
            base_offset: Additional base time for complex operations (e.g., structured extraction)
        """
        extra = (text_len / 1000) * self.TIMEOUT_PER_1K_CHARS
        timeout = self.BASE_TIMEOUT_SEC + base_offset + extra
        return min(timeout, self.MAX_TIMEOUT_SEC + base_offset)
    
    async def extract_claims(
        self,
        text: str,
        *,
        chunks: list[TextChunk] | None = None,  # M74
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
        
        # Updated Strategist Prompt with Salience & Harm Potential (M77) + Satire (M78)
        instructions = f"""You are an expert Fact-Checking Search Strategist.
Your goal is to extract verifiable claims AND develop optimal SEARCH STRATEGIES for each.

## NEGATIVE CONSTRAINTS (CRITICAL):
1. Do NOT generate queries seeking betting odds, gambling coefficients, or bookmaker predictions.
2. If the user asks for a prediction (e.g. sports), search for "expert analysis" or "official announcement", NOT "betting site".
3. EXCEPTION: If the user explicitly asks about "gambling regulations" or "corruption in betting", then betting terms ARE allowed.

## STEP 1: EXTRACT CLAIMS & ASSESS HARM (Impact-First)
For each claim in the article:
1. Extract the core factual assertion.
2. **SPLIT COMPOUND CLAIMS**:
   - If a sentence contains multiple distinct facts (e.g. "He did X, AND she did Y"), SPLIT them into separate claims!
   - Example: "The earthquake hit at 5 PM; tsunami followed at 6 PM" → Claim 1 (earthquake time), Claim 2 (tsunami time).
   - Do not combine verifiable facts into one "megaclaim" unless they are inseparable.

3. **Assess Harm Potential (1-5)**:
   - **5 (Critical)**: "Cures cancer", "Drink bleach", "Do not vaccinate", "Violent call to action". Immediate safety risk.
   - **4 (High)**: Medical/financial advice, "XYZ is toxic", political accusations without proof.
   - **3 (Medium)**: Controversial political statements, economic figures, historical revisionism.
   - **2 (Low)**: Attribution ("He said X"), timeline events ("Meeting held on Y").
   - **1 (Neutral)**: Definitions, physical constants ("Water boils at 100C"), background info.

4. **Prioritize Extraction**: Focus on Levels 4-5.
   - If an article has dangerous claims, extracted them FIRST.
   - Avoid filling slots with Level 1 trivia if Level 3+ claims exist.

## STEP 1.5: CLASSIFY CLAIM CATEGORY (Satire Detection)
For each claim, classify its **claim_category**:
- **FACTUAL**: Standard verifiable factual claim. Proceed with verification.
- **SATIRE**: Obvious parody, humor, absurdist claim (e.g., "Worms eating asphalt", "The Onion" style). 
  - Markers: Absurd scenarios, known satire sources, exaggerated humor, impossible events.
- **OPINION**: Subjective editorial opinion (e.g., "This policy is terrible").
- **HYPERBOLIC**: Exaggerated rhetoric not meant literally (e.g., "The best thing ever").

Also provide **satire_likelihood** (0.0-1.0):
- 0.0-0.3: Clearly factual
- 0.4-0.6: Ambiguous, could be exaggeration
- 0.7-0.8: Likely satire/opinion
- 0.9-1.0: Definitely satire (skip verification)

**ROUTING RULE**: If satire_likelihood >= 0.8, do NOT generate search queries. Mark as SATIRE.

## STEP 1.6: CLAIM METADATA FOR ORCHESTRATION (M80/M81)
For each claim, provide orchestration metadata:

1. **verification_target** (CRITICAL - READ CAREFULLY):
   - **"attribution"**: Use when the claim reports what someone SAID/STATED/CLAIMED.
     * MARKERS: "said", "stated", "claimed", "announced", "told", "revealed", "explained",
       "in an interview", "according to X", "X mentioned", "X recalled", "X shared"
     * For INTERVIEW articles: MOST claims should be "attribution"!
     * Verification = "Did person X actually say this?"
   - **"reality"**: Factual claim about external events/data that can be verified 
     INDEPENDENTLY of what someone said.
     * External events: "earthquake killed 50 people", "stock rose 5%"
     * Scientific facts: "vaccine is 95% effective" (needs study, not quote)
     * Historical facts: "WWII ended in 1945"
   - **"existence"**: Verify that a source/document/recording exists.
   - **"none"**: NOT VERIFIABLE:
     * Predictions/forecasts ("will happen", "expected to")
     * Horoscopes, astrology, subjective opinions

   ⚠️ ATTRIBUTION DETECTION RULE:
   If text contains phrases like "розповіла", "сказала", "заявила", "повідомив",
   "в інтерв'ю", "said", "told", "announced", "according to", "recalled"
   → DEFAULT to verification_target="attribution", NOT "reality"!

2. **claim_role** (STRICT LIMITS):
   - **"core"**: Central claim of the article. **MAXIMUM 2 per article!**
   - **"support"**: Evidence supporting a core claim.
   - **"context"**: Background info (not a fact to verify).
   - **"attribution"**: Quote attribution (what someone said).
   - **"meta"**: Info about the article/source itself.
   - **"aggregated"**: Summary from multiple sources.
   - **"subclaim"**: Subordinate detail.
   
   ⚠️ ROLE DISTRIBUTION RULE:
   For a 5-claim article: max 2 "core", rest must be "support"/"context"/"attribution".
   If ALL claims are "core", you are doing it WRONG!

3. **search_locale_plan**:
   - primary: Main search language ("en" for science, article language for local news)
   - fallback: Backup languages ["en", ...]

4. **retrieval_policy**:
   - channels_allowed: ["authoritative", "reputable_news", "local_media", "social", "low_reliability_web"]
   - For high harm_potential (>=4): use ONLY ["authoritative"]
   - For verification_target="none": use [] (no search needed)
   - For verification_target="attribution": prioritize ["reputable_news"] (interview sources)

5. **metadata_confidence** (CALIBRATED):
   - **"high"**: Claim has direct, checkable data points (dates, numbers, official records)
   - **"medium"**: Interview/quote-based claims without primary source access
   - **"low"**: Vague, context-dependent, or unverifiable without specialized sources
   
   ⚠️ CONFIDENCE RULE: Interview quotes = "medium" (no primary source), NOT "high"!


## STEP 2: THINK LIKE A SEARCH STRATEGIST (Chain of Thought)
For each **FACTUAL** claim, REASON about:

1. **Intent Analysis**: What type of claim is this?
   - {intents_str}

2. **Primary Authority**: Where would the ORIGINAL evidence exist?
   - Science/Medicine → Journals. Best in ENGLISH.
   - Local Politics → Local news. Best in LOCAL language.
   - Sports → Official sites. AVOID betting sites!

3. **Language Strategy**: 
   - Scientific facts → Search in ENGLISH
   - Local news → Search in LOCAL language ({lang_name})

4. **Search Method Selection**:
   - **"news"**: Recent events (last 30 days).
   - **"general_search"**: Evergreen facts, medical advice, history.
   - **"academic"**: specialized studies.

## STEP 3: GENERATE QUERY CANDIDATES
Generate 2-3 query candidates for each **FACTUAL** claim.

**ATTRIBUTION QUERY RULE:**
If `verification_target="attribution"` (someone said X):
- You MUST include attribution markers: "interview", "quote", "said", "transcript", "statement".
- If the source entity is known (e.g. "BBC", "CNN"), include it: "Kate Winslet BBC interview".
- DO NOT search for the fact itself as if it happened (e.g. "Did paparazzi follow Kate") -> Search for the STATEMENTS ("Kate Winslet interview paparazzi quotes").

**SPECIAL RULE for HEALTH/MEDICAL claims:**
If harm_potential >= 4, you MUST include:
1. **SAFETY/TOXICITY**: "entity + toxicity / poisoning / hazards / CDC / WHO"
2. **EFFICACY**: "entity + clinical evidence / cancer treatment / systematic review"

**Standard Roles:**
1. **CORE** (Required): Priority 1.0
2. **NUMERIC**: Priority 0.8
3. **ATTRIBUTION**: Priority 0.7
4. **LOCAL** (Optional): Priority 0.5

Also assign:
- **topic_group**: High-level category ({topics_str})
- **topic_key**: Use NEAREST MARKDOWN HEADING.

## STEP 4: CLASSIFY ARTICLE INTENT
- "news": Recent specific events.
- "evergreen": Medical, history, how-to. (Force "general_search")
- "official": Gov/Company announcements.
- "opinion" / "prediction".

## STEP 5: EVIDENCE NEED CLASSIFICATION
- "empirical_study", "guideline", "official_stats", "expert_opinion", "anecdotal", "news_report", "unknown"

## OUTPUT FORMAT

```json
{{
  "article_intent": "news",
  "claims": [
    {{
      "text": "Original text",
      "normalized_text": "Trump announced tariffs...",
      "type": "core",
      "claim_category": "FACTUAL",
      "satire_likelihood": 0.0,
      "topic_group": "Politics",
      "topic_key": "Trump Tariffs",
      "importance": 0.9,
      "check_worthiness": 0.9,
      "harm_potential": 3,
      "verification_target": "reality",
      "claim_role": "core",
      "search_locale_plan": {{
        "primary": "en",
        "fallback": ["en"]
      }},
      "retrieval_policy": {{
        "channels_allowed": ["authoritative", "reputable_news"],
        "use_policy_by_channel": {{"social": "lead_only", "low_reliability_web": "lead_only"}}
      }},
      "metadata_confidence": "high",
      "search_strategy": {{
        "intent": "official_statement",
        "reasoning": "Needs official confirmation",
        "best_language": "en"
      }},
      "query_candidates": [
        {{"text": "Trump China tariffs 2025", "role": "CORE", "score": 1.0}}
      ],
      "search_method": "news",
      "search_queries": ["Trump China tariffs"],
      "evidence_req": {{"needs_primary": true, "needs_2_independent": true}},
      "evidence_need": "news_report",
      "check_oracle": false
    }}
  ]
}}
```

**EVIDENCE NEED RULES:**
- If verification_target="attribution" → MUST use "primary_source" or "quote_verification".
- If verification_target="reality" (science/health) → "empirical_study" or "expert_opinion".
- If verification_target="reality" (news) → "news_report".

**EXAMPLE: HOROSCOPE (verification_target=none)**
```json
{{
  "text": "Водоліям сьогодні пощастить у фінансах",
  "verification_target": "none",
  "claim_role": "context",
  "claim_category": "OPINION",
  "satire_likelihood": 0.0,
  "check_worthiness": 0.1,
  "search_locale_plan": {{"primary": "en", "fallback": []}},
  "retrieval_policy": {{"channels_allowed": []}},
  "metadata_confidence": "high",
  "query_candidates": [],
  "search_queries": []
}}
```

**EXAMPLE: INTERVIEW ARTICLE (verification_target=attribution)**
For an article about Kate Winslet interview:
```json
{{
  "article_intent": "news",
  "claims": [
    {{
      "text": "Вінслет розповіла, що папараці стежили за нею після Титаніка",
      "normalized_text": "Kate Winslet said paparazzi followed her after Titanic",
      "type": "attribution",
      "verification_target": "attribution",
      "claim_role": "core",
      "metadata_confidence": "medium",
      "check_worthiness": 0.7,
      "search_queries": ["Kate Winslet interview paparazzi Titanic"]
    }},
    {{
      "text": "Акторка згадала, що їй радили схуднути",
      "normalized_text": "The actress mentioned being told to lose weight",
      "type": "attribution",
      "verification_target": "attribution",
      "claim_role": "support",
      "metadata_confidence": "medium",
      "check_worthiness": 0.5,
      "search_queries": ["Kate Winslet weight pressure interview"]
    }},
    {{
      "text": "Титанік заробив понад 2 мільярди доларів",
      "normalized_text": "Titanic earned over 2 billion dollars",
      "type": "numeric",
      "verification_target": "reality",
      "claim_role": "support",
      "metadata_confidence": "high",
      "check_worthiness": 0.8,
      "search_queries": ["Titanic box office revenue"]
    }}
  ]
}}
```
Note: 2 of 3 claims are "attribution" (what she said), only 1 is "reality" (box office data).

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""
        prompt = f"""Extract 3-{max_claims} atomic verifiable claims.
PRIORITIZE HARM: Find claims with harm_potential 4-5 (Health/Safety) first.
Ignore separate definitions (harm_potential 1) unless false/dangerous.

ARTICLE:
{text_excerpt}

Return the result in JSON format.
"""
        try:
            # M81: Updated cache key to force prompt refresh with calibration rules
            cache_key = f"claim_strategist_v6_{lang}"

            # M73.5: Dynamic timeout based on input size
            dynamic_timeout = self._calculate_timeout(len(text_excerpt))
            logger.debug("[Claims] Input: %d chars, timeout: %.1f sec", len(text_excerpt), dynamic_timeout)
            
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                cache_key=cache_key,
                timeout=dynamic_timeout,
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
                
                # M77: Harm Potential
                harm_potential = int(rc.get("harm_potential", 1))
                harm_potential = max(1, min(5, harm_potential)) # Clamp 1-5

                # M78: Claim Category & Satire Likelihood
                claim_category = rc.get("claim_category", "FACTUAL")
                if claim_category not in {"FACTUAL", "SATIRE", "OPINION", "HYPERBOLIC"}:
                    claim_category = "FACTUAL"
                
                satire_likelihood = float(rc.get("satire_likelihood", 0.0))
                satire_likelihood = max(0.0, min(1.0, satire_likelihood))  # Clamp 0-1

                # M80: Parse ClaimMetadata
                metadata = self._parse_claim_metadata(rc, lang, harm_potential, claim_category, satire_likelihood)

                # M62+: Extract search strategy if present
                strategy = rc.get("search_strategy", {})
                
                # M78+M80: Satire/Non-verifiable Routing - clear search data
                search_queries = rc.get("search_queries", [])
                query_candidates = rc.get("query_candidates", [])
                should_skip_search = (
                    satire_likelihood >= 0.8 or 
                    claim_category == "SATIRE" or
                    metadata.should_skip_search
                )
                if should_skip_search:
                    search_queries = []  # Skip search
                    query_candidates = []
                    reason = "satire" if satire_likelihood >= 0.8 else f"target={metadata.verification_target.value}"
                    logger.info("[M80] Skip search (%s): %s", reason, normalized[:50])
                
                c = Claim(
                    id=f"c{idx+1}",
                    text=rc.get("text", ""),
                    type=rc.get("type", "core"),  # type: ignore
                    importance=float(rc.get("importance", 0.5)),
                    evidence_requirement=req,
                    search_queries=search_queries,
                    check_oracle=bool(rc.get("check_oracle", False)),
                    # M62: New fields
                    normalized_text=normalized,
                    topic_group=topic,
                    check_worthiness=worthiness,
                    # M64: New fields
                    topic_key=topic_key,
                    query_candidates=query_candidates,
                    # M66: Smart Routing
                    search_method=rc.get("search_method", "general_search"),
                    # M73 Layer 4: Evidence-Need Routing
                    evidence_need=rc.get("evidence_need", "unknown"),
                    # M74: Anchor
                    anchor=self._locate_anchor(rc.get("text", ""), chunks),
                    # M77: Salience
                    harm_potential=harm_potential,
                    # M78: Satire
                    claim_category=claim_category,
                    satire_likelihood=satire_likelihood,
                    # M80: Orchestration Metadata
                    metadata=metadata,
                )
                
                # Log strategy for debugging
                if strategy:
                    intent = strategy.get("intent", "?")
                    reasoning = strategy.get("reasoning", "")[:50]
                    logger.debug(
                        "[Strategist] Claim %d: intent=%s, harm=%d, category=%s | %s",
                        idx+1, intent, harm_potential, claim_category, reasoning
                    )
                
                claims.append(c)
            
            # ─────────────────────────────────────────────────────────────────
            # M73.5: DEDUPLICATION - Merge claims with identical normalized_text
            # ─────────────────────────────────────────────────────────────────
            claims = self._dedupe_claims(claims)

            # M77: Sort by harm_potential DESC
            claims.sort(key=lambda x: x.get("harm_potential", 1), reverse=True)
            
            # M78: Count satire claims for telemetry
            satire_count = sum(1 for c in claims if c.get("satire_likelihood", 0) >= 0.8 or c.get("claim_category") == "SATIRE")
            if satire_count > 0:
                logger.info("[M78] Detected %d satire/hyperbolic claims", satire_count)
            
            # Log topic and strategy distribution
            topics_found = [c.get("topic_key", "?") for c in claims]
            logger.info("[Claims] Extracted %d claims (after dedup/sort). Topics keys: %s", len(claims), topics_found)
                
            # M60 Oracle Optimization: Check if ANY claim needs oracle
            check_oracle = any(c.get("check_oracle", False) for c in claims)
            
            # M63: Log intent for debugging
            logger.info("[Claims] Article intent: %s (check_oracle=%s)", article_intent, check_oracle)
            
            # M81/T4: Trace extracted claims for debugging
            self._trace_extracted_claims(claims)
            
            # M80/T8: Trace metadata distribution for debugging
            self._trace_metadata_distribution(claims)
            
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
        chunks: list[TextChunk] | None = None,  # M74
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

            # M73.5: Dynamic timeout with +10s offset for structured extraction complexity
            dynamic_timeout = self._calculate_timeout(len(text_excerpt), base_offset=10.0)
            logger.debug("[Claims Structured] Input: %d chars, timeout: %.1f sec", len(text_excerpt), dynamic_timeout)
            
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="medium",  # Higher effort for schema parsing
                cache_key=cache_key,
                timeout=dynamic_timeout,
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

    # ─────────────────────────────────────────────────────────────────────────
    # M73.5: Claim Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def _locate_anchor(self, text: str, chunks: list[TextChunk] | None) -> ClaimAnchor | None:
        """M74: Locate claim anchor in text chunks."""
        if not text or not chunks:
            return None
        
        # Simple exact substring match of first 100 chars (case-insensitive)
        # Sufficient for anchoring
        target = " ".join(text.lower().split())[:100]
        
        for ch in chunks:
            if target in ch.text.lower():
                 start = ch.text.lower().find(target)
                 return {
                     "chunk_id": ch.chunk_id,
                     "char_start": ch.char_start + start,
                     "char_end": ch.char_start + start + len(text),
                     "section_path": ch.section_path
                 }
        return None

    def _dedupe_claims(self, claims: list[Claim]) -> list[Claim]:
        """
        Deduplicate claims by normalized_text hash.
        
        Merges duplicates:
        - Takes MAX importance (most important wins)
        - Takes MAX check_worthiness
        - Keeps first occurrence (for type, topic_group, etc.)
        - Merges search_queries and query_candidates
        
        M74: Spatial Deduplication
        - Uses simple bucketing (200 chars) to determine if claims are distinct instances
        """
        if not claims:
            return []
        
        # Group by normalized_text (lowercased, stripped)
        seen: dict[str, Claim] = {}
        
        for c in claims:
            # Normalize key: lowercase, strip, collapse whitespace
            base_key = " ".join((c.get("normalized_text") or c.get("text") or "").lower().split())
            if not base_key:
                continue
            
            # M74: Spatial bucketing
            key = base_key
            if c.get("anchor"):
                # Claims >200 chars apart are treated as separate
                bucket = c["anchor"]["char_start"] // 200
                key = f"{base_key}|loc:{bucket}"
            
            if key in seen:
                # Merge: take max importance
                existing = seen[key]
                existing["importance"] = max(
                    float(existing.get("importance", 0.5)),
                    float(c.get("importance", 0.5))
                )
                existing["check_worthiness"] = max(
                    float(existing.get("check_worthiness", 0.5)),
                    float(c.get("check_worthiness", 0.5))
                )
                # Merge query candidates (dedupe by text)
                existing_queries = existing.get("query_candidates", []) or []
                new_queries = c.get("query_candidates", []) or []
                seen_texts = {q.get("text") for q in existing_queries if q}
                for q in new_queries:
                    if q and q.get("text") not in seen_texts:
                        existing_queries.append(q)
                        seen_texts.add(q.get("text"))
                existing["query_candidates"] = existing_queries
                # Oracle: if ANY duplicate wants oracle, check it
                if c.get("check_oracle"):
                    existing["check_oracle"] = True
                logger.debug("[Dedup] Merged claim: %s", key[:50])
            else:
                seen[key] = c
        
        # Re-assign IDs (c1, c2, ...)
        deduped = list(seen.values())
        for idx, c in enumerate(deduped):
            c["id"] = f"c{idx + 1}"
        
        # Log dedup stats
        if len(claims) != len(deduped):
            logger.info("[Dedup] Merged %d → %d claims", len(claims), len(deduped))
        
        return deduped

    # ─────────────────────────────────────────────────────────────────────────
    # M80: Metadata Parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_claim_metadata(
        self,
        rc: dict,
        lang: str,
        harm_potential: int,
        claim_category: str,
        satire_likelihood: float,
    ) -> ClaimMetadata:
        """
        M80: Parse Claim metadata from LLM response with safe fallback defaults.

        Delegated to `spectrue_core.agents.skills.claim_metadata_parser.parse_claim_metadata`
        to keep this file readable.
        """
        return parse_claim_metadata_v1(
            rc,
            lang=lang,
            harm_potential=harm_potential,
            claim_category=claim_category,
            satire_likelihood=satire_likelihood,
        )

    def _trace_extracted_claims(self, claims: list[Claim]) -> None:
        """
        M81/T4: Emit trace event with individual claim details for debugging.
        
        Logs first 100 chars of each claim text with metadata.
        Critical for diagnosing attribution/reality misclassification.
        """
        if not claims:
            return
        
        claims_data = []
        for c in claims[:7]:  # Max 7 claims
            metadata = c.get("metadata")
            claims_data.append({
                "id": c.get("id", "?"),
                "text": c.get("text", "")[:100],  # First 100 chars
                "verification_target": metadata.verification_target.value if metadata else "?",
                "claim_role": metadata.claim_role.value if metadata else "?",
                "check_worthiness": c.get("check_worthiness", 0),
                "metadata_confidence": metadata.metadata_confidence.value if metadata else "?",
            })
        
        Trace.event("claim_extraction.claims_extracted", {
            "count": len(claims),
            "claims": claims_data,
        })

    def _trace_metadata_distribution(self, claims: list[Claim]) -> None:
        """
        M80/T8: Emit trace event with metadata distribution for debugging.
        
        Logs distribution counts for:
        - verification_target: {reality: N, attribution: M, existence: K, none: L}
        - claim_role: {core: N, support: M, context: K, ...}
        - metadata_confidence: {low: N, medium: M, high: K}
        """
        if not claims:
            return
        
        # Count verification_target distribution
        target_dist: dict[str, int] = {"reality": 0, "attribution": 0, "existence": 0, "none": 0}
        role_dist: dict[str, int] = {}
        confidence_dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        skip_search_count = 0
        
        for claim in claims:
            metadata = claim.get("metadata")
            if metadata:
                # Target distribution
                target = metadata.verification_target.value
                target_dist[target] = target_dist.get(target, 0) + 1
                
                # Role distribution
                role = metadata.claim_role.value
                role_dist[role] = role_dist.get(role, 0) + 1
                
                # Confidence distribution
                confidence = metadata.metadata_confidence.value
                confidence_dist[confidence] = confidence_dist.get(confidence, 0) + 1
                
                # Skip search count
                if metadata.should_skip_search:
                    skip_search_count += 1
        
        # Emit trace event
        Trace.event("claim_extraction.metadata_distribution", {
            "total_claims": len(claims),
            "verification_target": target_dist,
            "claim_role": role_dist,
            "metadata_confidence": confidence_dist,
            "skip_search_count": skip_search_count,
        })
        
        # Also log summary
        logger.info(
            "[M80] Metadata: targets=%s, roles=%s, confidence=%s, skip_search=%d",
            target_dist, role_dist, confidence_dist, skip_search_count
        )

    def _default_channels(
        self,
        harm_potential: int,
        verification_target: VerificationTarget,
    ) -> list[EvidenceChannel]:
        return default_channels_v1(
            harm_potential=harm_potential,
            verification_target=verification_target,
        )
