# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX


def build_claim_strategist_instructions(*, intents_str: str, topics_str: str, lang_name: str) -> str:
    return f"""**OUTPUT RULE: Output ONLY valid JSON. No prose. No prefixes. No markdown. Start with {{**

You are an expert Fact-Checking Search Strategist.
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

## STEP 1.6: CLAIM METADATA FOR ORCHESTRATION 
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
   - **"thesis"**: Main thesis or conclusion. (Use sparingly for the central points).
   - **"support"**: Evidence supporting a thesis claim.
   - **"background"**: Background context (explain-only).
   - **"example"**: Illustrative example for another claim.
   - **"hedge"**: Qualified/uncertain statement ("may", "might").
   - **"counterclaim"**: Opposing or rebuttal claim.
   
   ⚠️ ROLE DISTRIBUTION:
   Prioritize "thesis" for the central arguments. Use "support" for the specific evidence backing them.
   If EVERYTHING is a "thesis", you are doing it WRONG!

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

## STEP 1.7: CLAIM STRUCTURE (M93)
For each claim, provide a `structure` object:
- **type**: empirical_numeric | event | causal | attribution | definition | policy_plan | forecast | meta_scientific
- **premises**: list of premise statements (can be empty)
- **conclusion**: the conclusion statement (usually the main claim text)
- **dependencies**: list of claim IDs (e.g., ["c1", "c3"]) that must be verified first

Only add dependencies when the conclusion logically depends on other claims.

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

IMPORTANT: Every claim MUST include all keys shown below. Use empty arrays/objects
or neutral defaults when a field does not apply.
- "text" MUST be an exact substring from the article (no paraphrase).
- **CRITICAL**: "text" must be in the SAME LANGUAGE as the article. Do NOT translate to English!
  - If article is in Ukrainian, "text" MUST be in Ukrainian (exact quote from article)
  - If article is in Russian, "text" MUST be in Russian
  - "normalized_text" can be in English for search purposes, but "text" is ALWAYS original language

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
      "claim_role": "thesis",
      "structure": {{
        "type": "policy_plan",
        "premises": ["Policy X targets imports from China"],
        "conclusion": "Trump announced tariffs on China in 2025",
        "dependencies": ["c2"]
      }},
      "search_locale_plan": {{
        "primary": "en",
        "fallback": ["en"]
      }},
      "retrieval_policy": {{
        "channels_allowed": ["authoritative", "reputable_news"]
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
  "article_intent": "opinion",
  "claims": [
    {{
      "text": "Водоліям сьогодні пощастить у фінансах",
      "normalized_text": "Aquarius will have financial luck today",
      "type": "opinion",
      "claim_category": "OPINION",
      "satire_likelihood": 0.0,
      "topic_group": "Other",
      "topic_key": "Horoscope",
      "importance": 0.2,
      "check_worthiness": 0.1,
      "harm_potential": 1,
      "verification_target": "none",
      "claim_role": "background",
      "structure": {{
        "type": "forecast",
        "premises": [],
        "conclusion": "Водоліям сьогодні пощастить у фінансах",
        "dependencies": []
      }},
      "search_locale_plan": {{"primary": "uk", "fallback": ["en"]}},
      "retrieval_policy": {{"channels_allowed": []}},
      "metadata_confidence": "high",
      "search_strategy": {{
        "intent": "prediction_opinion",
        "reasoning": "Horoscope-style statement without verifiable source",
        "best_language": "uk"
      }},
      "query_candidates": [],
      "search_method": "general_search",
      "search_queries": [],
      "evidence_req": {{"needs_primary": false, "needs_2_independent": false}},
      "evidence_need": "unknown",
      "check_oracle": false
    }}
  ]
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
      "claim_category": "FACTUAL",
      "satire_likelihood": 0.0,
      "topic_group": "Culture",
      "topic_key": "Kate Winslet Interview",
      "importance": 0.7,
      "check_worthiness": 0.7,
      "harm_potential": 2,
      "verification_target": "attribution",
      "claim_role": "thesis",
      "structure": {{
        "type": "attribution",
        "premises": [],
        "conclusion": "Kate Winslet said paparazzi followed her after Titanic",
        "dependencies": []
      }},
      "search_locale_plan": {{"primary": "en", "fallback": ["uk"]}},
      "retrieval_policy": {{"channels_allowed": ["reputable_news", "authoritative"]}},
      "metadata_confidence": "medium",
      "search_strategy": {{
        "intent": "quote_attribution",
        "reasoning": "Interview statement requires primary source",
        "best_language": "en"
      }},
      "query_candidates": [
        {{"text": "Kate Winslet interview paparazzi after Titanic", "role": "CORE", "score": 1.0}}
      ],
      "search_method": "news",
      "search_queries": ["Kate Winslet interview paparazzi Titanic"],
      "evidence_req": {{"needs_primary": true, "needs_2_independent": false}},
      "evidence_need": "news_report",
      "check_oracle": false
    }},
    {{
      "text": "Акторка згадала, що їй радили схуднути",
      "normalized_text": "The actress mentioned being told to lose weight",
      "type": "attribution",
      "claim_category": "FACTUAL",
      "satire_likelihood": 0.0,
      "topic_group": "Culture",
      "topic_key": "Kate Winslet Interview",
      "importance": 0.5,
      "check_worthiness": 0.5,
      "harm_potential": 2,
      "verification_target": "attribution",
      "claim_role": "support",
      "structure": {{
        "type": "attribution",
        "premises": [],
        "conclusion": "The actress mentioned being told to lose weight",
        "dependencies": []
      }},
      "search_locale_plan": {{"primary": "en", "fallback": ["uk"]}},
      "retrieval_policy": {{"channels_allowed": ["reputable_news", "authoritative"]}},
      "metadata_confidence": "medium",
      "search_strategy": {{
        "intent": "quote_attribution",
        "reasoning": "Interview statement requires primary source",
        "best_language": "en"
      }},
      "query_candidates": [
        {{"text": "Kate Winslet told to lose weight interview", "role": "SUPPORT", "score": 0.8}}
      ],
      "search_method": "news",
      "search_queries": ["Kate Winslet weight pressure interview"],
      "evidence_req": {{"needs_primary": true, "needs_2_independent": false}},
      "evidence_need": "news_report",
      "check_oracle": false
    }},
    {{
      "text": "Титанік заробив понад 2 мільярди доларів",
      "normalized_text": "Titanic earned over 2 billion dollars",
      "type": "numeric",
      "claim_category": "FACTUAL",
      "satire_likelihood": 0.0,
      "topic_group": "Culture",
      "topic_key": "Titanic Box Office",
      "importance": 0.8,
      "check_worthiness": 0.8,
      "harm_potential": 2,
      "verification_target": "reality",
      "claim_role": "support",
      "structure": {{
        "type": "empirical_numeric",
        "premises": [],
        "conclusion": "Titanic earned over 2 billion dollars",
        "dependencies": []
      }},
      "search_locale_plan": {{"primary": "en", "fallback": ["uk"]}},
      "retrieval_policy": {{"channels_allowed": ["authoritative", "reputable_news"]}},
      "metadata_confidence": "high",
      "search_strategy": {{
        "intent": "historical_event",
        "reasoning": "Box office totals require reputable sources",
        "best_language": "en"
      }},
      "query_candidates": [
        {{"text": "Titanic box office revenue total", "role": "NUMERIC", "score": 0.9}}
      ],
      "search_method": "news",
      "search_queries": ["Titanic box office revenue"],
      "evidence_req": {{"needs_primary": false, "needs_2_independent": true}},
      "evidence_need": "official_stats",
      "check_oracle": false
    }}
  ]
}}
```
Note: 2 of 3 claims are "attribution" (what she said), only 1 is "reality" (box office data).

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""


def build_claim_strategist_prompt(text_excerpt: str) -> str:
    return f"""Tasks:
1. Analyze the article text below.
2. Extract ALL distinct, atomic, check-worthy factual assertions (claims).
   - Do NOT limit the number of claims. Extract everything that matters.
   - Ignore trivial details or filler text.
   - Separate compound sentences into individual atomic claims.
3. For each claim, provide the full metadata as defined in the system instructions.

ARTICLE:
{text_excerpt}

Return the result in JSON format.
"""


def build_claim_schema_instructions(*, topics_str: str, lang_name: str) -> str:
    return f"""You are a SCHEMA PARSER for fact-checking.

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
  "article_intent": "news|evergreen|official|opinion|prediction|unknown|other",
  "claims": [
    {{
      "id": "c1",
      "claim_type": "event|attribution|numeric|definition|timeline|other",
      "claim_role": "thesis|support|background|example|hedge|counterclaim",
      "structure": {{
        "type": "empirical_numeric|event|causal|attribution|definition|policy_plan|forecast|meta_scientific",
        "premises": [],
        "conclusion": "Short conclusion sentence",
        "dependencies": []
      }},
      
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


def build_claim_schema_prompt(*, text_excerpt: str) -> str:
    return f"""Parse this article into structured claims.
For each claim, identify FACT assertions (verifiable) and CONTEXT assertions (informational).

CRITICAL: Time zone references like "(в Україні)", "(за Києвом)" are CONTEXT, not location!

ARTICLE:
{text_excerpt}

Return structured ClaimUnits in JSON format.
"""

def build_core_extraction_prompt(*, text_excerpt: str) -> str:
    """
    Verifiability-first core extraction prompt.
    Enforces structural anchors and filters out non-verifiable content.
    """
    return f"""**OUTPUT RULE: Output ONLY valid JSON. No prose. No prefixes. No markdown. Start with {{**

You are extracting **externally verifiable factual propositions** from text.

## VERIFIABILITY CONTRACT (CRITICAL)

A claim is verifiable if and only if it has:
1. **Subject entities**: At least one named or scientific entity (person, organization, place, product, chemical, biological entity, or specific natural object like "Crystals", "DNA", "Solar Wind")
2. **Time anchor**: When it happened/was stated OR evidence of being a "timeless truth" (natural law, physical property, mathematical definition)
3. **Falsifiability**: The claim can be proven TRUE or FALSE using external evidence (scientific journals, official stats, textbooks, reputable news)

## REQUIRED FIELDS FOR EACH CLAIM

For every claim you extract, provide:

1. **claim_text**: Exact substring from the article (original language)
2. **normalized_text**: Self-sufficient English summary for search
3. **subject_entities**: List of canonical entity names (STRICTLY 1-5 items, REQUIRED)
   - Do NOT list every mention. Include ONLY the TOP 5 most relevant entities.
   - Example: ["Elon Musk", "Tesla", "SEC"]
4. **predicate_type**: One of: "event", "measurement", "policy", "quote", "ranking", "causal", "existence", "definition", "property", "other"
5. **time_anchor**: Object with {{"type": "explicit_date"|"range"|"relative"|"timeless"|"unknown", "value": "<extracted or unknown>"}}
6. **location_anchor**: Geographic context or "unknown"
7. **falsifiability**: Object with:
   - is_falsifiable: boolean (MUST be true for claims you emit)
   - falsifiable_by: one of "public_records", "scientific_publication", "official_statement", "reputable_news", "dataset", "other"
8. **expected_evidence**: Object with {{"evidence_kind": "primary_source"|"secondary_source"|"both", "likely_sources": [...]}}
9. **retrieval_seed_terms**: Array of 3-10 KEYWORDS (not sentences!) derived from entities + key noun phrases
   - STRICTLY LIMITED TO 10 terms. Choose the most specific ones.
   - GOOD: ["Tesla", "SEC", "fraud", "settlement", "2024"]
   - BAD: ["Tesla was sued by the SEC for fraud in 2024"]
10. **importance**: Float 0.0-1.0 for prioritization

## DO NOT EMIT (drop silently)

❌ **Opinions without facts**: "This is an encouraging sign", "The policy is terrible"
❌ **Meta-statements**: "There is insufficient evidence", "This does not prove anything"
❌ **Rhetorical summaries**: "In conclusion, X is important"
❌ **Unanchored generalizations**: "People tend to...", "Life is complex" (vague social generalizations without specific entity or metric)
   - *Note*: Scientific definitions like "Water boils at 100C" are NOT generalizations; they are verifiable properties.
❌ **Predictions/forecasts**: "X will happen", "Expected to increase"
❌ **Subjective evaluations**: "X is the best", "Y is undervalued" (without metric)

## TRANSFORM OR DROP

If you encounter vague content, either:
1. **Transform** into a verifiable claim by finding anchors:
   - VAGUE: "Unemployment has increased"
   - VERIFIABLE: "US unemployment rate rose to 4.3% in July 2024" (with entities, time, measurement)
2. **Drop** if no anchors can be extracted from context

## EXISTENCE/EVALUATION HANDLING

Do NOT emit generic "existence" claims like "X exists" or "Y is good".
Instead, transform into:
- **Event claims**: "X was founded in 2010 in California"
- **Measurement claims**: "X has 500 employees as of Q4 2024"

If you cannot provide entity + time anchors, do NOT emit the claim.

## OUTPUT FORMAT

```json
{{
  "article_intent": "news|evergreen|opinion|official|prediction|unknown|other",
  "claims": [
    {{
      "claim_text": "Original language exact quote",
      "normalized_text": "Self-sufficient English version",
      "subject_entities": ["Entity1", "Entity2"],
      "predicate_type": "event",
      "time_anchor": {{"type": "explicit_date", "value": "January 5, 2025"}},
      "location_anchor": "Washington, DC",
      "falsifiability": {{
        "is_falsifiable": true,
        "falsifiable_by": "reputable_news"
      }},
      "expected_evidence": {{
        "evidence_kind": "secondary_source",
        "likely_sources": ["reputable_news", "authoritative"]
      }},
      "retrieval_seed_terms": ["Entity1", "Entity2", "key", "terms", "here"],
      "importance": 0.85
    }}
  ]
}}
```

## QUALITY OVER QUANTITY

Extract ALL important verifiable claims, but ONLY those that pass the verifiability contract.
It is better to extract 3 high-quality verifiable claims than 10 weak/vague claims.

ARTICLE:
{text_excerpt}

Return JSON with list of verifiable claims only (or empty array if none found).
"""


def build_retrieval_planning_prompt(
    claim_text: str,
    article_context_sm: str,
    lang_name: str,
) -> str:
    # article_context_sm should be a smaller/summarized version or just the full chunk if it fits.
    # We will use the full chunk for now as we want deep context.
    return f"""**CRITICAL OUTPUT RULES:**
1. Output ONLY a single JSON object with retrieval-planning fields.
2. Do NOT wrap in {{"claims": [...]}} or {{"article_intent": ...}}.
3. Do NOT include "text" or "normalized_text" fields - those are already known.
4. Start your response with {{ and end with }}.

You are planning retrieval metadata for ONE specific claim that was already extracted.

CLAIM TO PLAN: "{claim_text}"

ARTICLE CONTEXT:
{article_context_sm}

**CRITICAL: search_queries FORMAT REQUIREMENTS:**
- MUST be a non-empty array with 1-5 keyword queries (STRICTLY LIMITED to TOP 5)
- Each query: 2-8 words, MAX 80 characters
- Format: keyword phrases ONLY (NOT full sentences)
- NO trailing periods or punctuation
- Prefer "news" as search_method unless content is evergreen/academic

**GOOD search_queries examples:**
- ["Ukraine military offensive Kherson", "Zelenskyy statement troops"]
- ["COVID vaccine efficacy Pfizer study"]
- ["Tesla stock price Q4 2025"]

**BAD search_queries examples (DO NOT USE):**
- ["The president announced new policy yesterday."] (full sentence, has period)
- ["What is the effectiveness of vaccines?"] (question format)
- [] (empty array, NEVER do this)

**Required fields in your JSON response:**
- claim_category: "FACTUAL" | "OPINION" | "SATIRE" | "HYPERBOLIC"
- harm_potential: 1-5 (1=low, 5=critical)
- verification_target: "reality" | "attribution" | "existence" | "none"
- claim_role: "core" | "thesis" | "support" | "background" | "context" | "meta" | "attribution" | "aggregated" | "subclaim" | "example" | "hedge" | "counterclaim" | "definition" | "forecast"
- satire_likelihood: 0.0-1.0
- importance: 0.0-1.0
- check_worthiness: 0.0-1.0
- structure: {{"type": "empirical_numeric"|"event"|"causal"|"attribution"|"definition"|"policy_plan"|"forecast"|"meta_scientific"|"other", "premises": [], "conclusion": "...", "dependencies": []}}
- search_locale_plan: {{"primary": "{lang_name[:2].lower()}", "fallback": ["en"]}}
- retrieval_policy: {{"channels_allowed": [...]}} where ONLY these values are allowed: "authoritative", "reputable_news", "local_media", "social", "low_reliability_web" (NOT "academic"!)
- metadata_confidence: "low" | "medium" | "high"
- query_candidates: [{{"text": "...", "role": "CORE", "score": 1.0}}]
- search_method: "news" | "general_search" | "academic" (DEFAULT to "news" for recent events)
- search_queries: ["keyword query 1", "keyword query 2"] (REQUIRED, non-empty, 2-8 words each)
- evidence_req: {{"needs_primary": true/false, "needs_2_independent": true/false}}
- evidence_need: "empirical_study" | "guideline" | "official_stats" | "expert_opinion" | "anecdotal" | "news_report" | "unknown"
- check_oracle: true/false

Output the JSON object now (no markdown, no wrapper):
"""


def build_metadata_enrichment_prompt(
    *,
    claim_text: str,
    article_context_sm: str,
    intents_str: str,
    topics_str: str,
    lang_name: str,
) -> str:
    return build_retrieval_planning_prompt(
        claim_text=claim_text,
        article_context_sm=article_context_sm,
        intents_str=intents_str,
        topics_str=topics_str,
        lang_name=lang_name,
    )


def build_skeleton_extraction_prompt(*, text_excerpt: str) -> str:
    """
    Coverage Skeleton extraction prompt.
    
    Phase-1: Extract ALL events/measurements/quotes/policies with raw_span.
    No filtering at this stage - aim for complete coverage.
    """
    return f"""**OUTPUT RULE: Output ONLY valid JSON. No prose. No prefixes. No markdown. Start with {{**

You are extracting a **COVERAGE SKELETON** from text - identifying ALL factual elements.

## YOUR TASK

Extract EVERY verifiable element into one of four categories:
1. **Events**: Things that happened (actions, changes, occurrences)
2. **Measurements**: Numbers, statistics, quantities with metrics
3. **Quotes**: Statements attributed to specific speakers
4. **Policies**: Rules, regulations, decisions by authorities

## CRITICAL: COMPLETE COVERAGE

Your goal is **100% recall** - capture EVERYTHING, not just "main" claims.
- If there is a number in the text → there should be a measurement
- If someone said something → there should be a quote
- If something happened → there should be an event
- If a rule/policy is mentioned → there should be a policy

## REQUIRED FIELDS

For EVERY item you extract, include:
- **id**: Unique identifier (evt_1, msr_1, qot_1, pol_1)
- **subject_entities**: List of entity names involved
- **raw_span**: EXACT substring from the input (for anchoring)

### Events Schema
{{
  "id": "evt_1",
  "subject_entities": ["Entity1", "Entity2"],
  "verb_phrase": "announced partnership with",
  "time_anchor": {{"type": "explicit_date", "value": "January 2025"}},
  "location_anchor": "New York",
  "raw_span": "Entity1 announced partnership with Entity2 in New York in January 2025"
}}

### Measurements Schema  
{{
  "id": "msr_1",
  "subject_entities": ["Company"],
  "metric": "revenue",
  "quantity_mentions": [{{"value": "5.2", "unit": "billion dollars"}}],
  "time_anchor": {{"type": "explicit_date", "value": "Q4 2024"}},
  "raw_span": "Company reported revenue of $5.2 billion in Q4 2024"
}}

### Quotes Schema
{{
  "id": "qot_1",
  "speaker_entities": ["CEO Name"],
  "quote_text": "We expect continued growth",
  "raw_span": "CEO Name said \"We expect continued growth\""
}}

### Policies Schema
{{
  "id": "pol_1",
  "subject_entities": ["European Union"],
  "policy_action": "requires all cars to have emergency braking",
  "time_anchor": {{"type": "explicit_date", "value": "2025"}},
  "raw_span": "The EU requires all new cars to have emergency braking from 2025"
}}

## TIME_ANCHOR TYPES
- "explicit_date": YYYY, MM-YYYY, DD-MM-YYYY, "Q4 2024", "January 2025"
- "range": "between 2020 and 2023", "from January to March"
- "relative": "yesterday", "last week", "recently"
- "unknown": No time reference found (use null for value)

## OUTPUT FORMAT

```json
{{
  "events": [
    {{"id": "evt_1", "subject_entities": [...], "verb_phrase": "...", "time_anchor": {...}, "location_anchor": "...", "raw_span": "..."}}
  ],
  "measurements": [
    {{"id": "msr_1", "subject_entities": [...], "metric": "...", "quantity_mentions": [...], "time_anchor": {...}, "raw_span": "..."}}
  ],
  "quotes": [
    {{"id": "qot_1", "speaker_entities": [...], "quote_text": "...", "raw_span": "..."}}
  ],
  "policies": [
    {{"id": "pol_1", "subject_entities": [...], "policy_action": "...", "time_anchor": {...}, "raw_span": "..."}}
  ]
}}
```

## EXTRACTION RULES

1. **One element per item**: Don't combine multiple facts into one
2. **Raw span precision**: The raw_span must be an EXACT substring from input
3. **Entity canonicalization**: Use full proper names ("Elon Musk" not "he")
4. **Numbers → Measurements**: ANY numeric value should create a measurement
5. **Speeches → Quotes**: ANY attributed statement should create a quote

ARTICLE:
{text_excerpt}

Return JSON with all extracted skeleton items (empty arrays are OK for unused categories).
"""
