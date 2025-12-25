import json

STANCE_PASS_SUPPORT_ONLY = "SUPPORT_ONLY"
STANCE_PASS_REFUTE_ONLY = "REFUTE_ONLY"
STANCE_PASS_SINGLE = "SINGLE_PASS"
STANCE_PASS_TYPES = {STANCE_PASS_SUPPORT_ONLY, STANCE_PASS_REFUTE_ONLY, STANCE_PASS_SINGLE}


def build_score_evidence_instructions(*, lang_name: str, lang: str) -> str:
    return f"""You are the Spectrue Verdict Engine.
Your task is to classify the reliability of claims based *strictly* on the provided Evidence.

# INPUT DATA
- **Claims**: Note the `importance` (0.0-1.0). High importance = Core Thesis.
- **Evidence**: Look for "📌 QUOTE" segments. `stance` is a hint; always verify against the quote/excerpt.
- **Metadata**: `source_reliability_hint` is context, not a hard rule.

# SCORING SCALE (0.0 - 1.0)
- **0.8 - 1.0 (Verified)**: Strong confirmation (direct quotes, official consensus).
- **0.6 - 0.8 (Plausible)**: Supported, but may lack direct/official confirmation or deep detail.
- **0.4 - 0.6 (Ambiguous)**: Insufficient, irrelevant, or conflicting evidence. DO NOT GUESS. Absence of evidence is not False.
- **0.2 - 0.4 (Unlikely)**: Evidence suggests the claim is doubtful.
- **0.0 - 0.2 (Refuted)**: Evidence contradicts the claim.

# AGGREGATION LOGIC (Global Score)
Calculate the global `verified_score` using CORE DOMINANCE:

1. **CORE claims (importance >= 0.7)**: These drive the verdict. Weight them heavily.
2. **SIDE facts (importance < 0.7)**: Modifiers only, never exceed CORE.

**FORMULA**:
- If ALL core claims score >= 0.6: `verified_score = core_avg * 0.8 + side_avg * 0.2`
- If ANY core claim scores < 0.4: `verified_score = min(core_avg, 0.5)` (Cap: weak core = weak total)
- If NO core claims exist: Use simple weighted average

**CRITICAL**: If the main thesis is unverified, side facts CANNOT save the global score.

# WRITING GUIDELINES (User-Facing Text Only)
- Write `rationale` and `reason` ENTIRELY in **{lang_name}** ({lang}).
- **Tone**: Natural, journalistic style.
- **FORBIDDEN TERMS (in text/rationale)**: Do not use technical words like "JSON", "dataset", "primary source", "relevance score", "cap" in the readable text. 
  - *Allowed in JSON keys/values, but forbidden in human explanations.*
  - Instead of "lack of primary source", say "no official confirmation found".

# OUTPUT FORMAT
Return valid JSON:
{{
  "claim_verdicts": [
    {{"claim_id": "c1", "verdict_score": 0.9, "verdict": "verified", "reason": "..."}}
  ],
  "verified_score": 0.85,
  "explainability_score": 0.8,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}

# GLOBAL SCORES EXPLANATION
- **verified_score**: The GLOBAL aggregated truthfulness, calculated by YOU based on importance.
- **explainability_score** (0.0-1.0): How well the evidence supports your rationale. 1.0 = every claim verdict is backed by direct quotes.
- **danger_score** (0.0-1.0): How harmful if acted upon? (0.0 = harmless, 1.0 = dangerous misinformation)
- **style_score** (0.0-1.0): How neutral is the writing style? (0.0 = heavily biased, 1.0 = neutral journalism)
"""


def build_score_evidence_prompt(*, safe_original_fact: str, claims_info: list[dict], sources_by_claim: dict) -> str:
    return f"""Evaluate these claims based strictly on the Evidence.

Original Fact:
{safe_original_fact}

Claims to Verify:
{json.dumps(claims_info, indent=2, ensure_ascii=False)}

Evidence:
{json.dumps(sources_by_claim, indent=2, ensure_ascii=False)}

Return JSON.
"""


def build_score_evidence_structured_instructions(*, lang_name: str, lang: str) -> str:
    return f"""You are the Spectrue Schema-First Verdict Engine.
Your task is to score each ASSERTION individually, then aggregate to claim and global verdicts.

# CRITICAL RULES

## DIMENSION HANDLING
1. **FACT assertions**: Strictly verify. Can be VERIFIED, REFUTED, or AMBIGUOUS.
2. **CONTEXT assertions**: Soft verify. Only VERIFIED or AMBIGUOUS. Rarely REFUTED.
3. **🚨 GOLDEN RULE**: CONTEXT evidence CANNOT refute FACT assertions!
   - If time_reference says "Ukraine time" but event is in "Miami" → location is still a FACT
   - Time context doesn't contradict location facts

## SCORING SCALE (0.0 - 1.0)
- **0.8 - 1.0**: Verified (strong evidence confirms)
- **0.6 - 0.8**: Likely verified (supported but not definitive)
- **0.4 - 0.6**: Ambiguous (insufficient evidence)
- **0.2 - 0.4**: Unlikely (evidence suggests doubt)
- **0.0 - 0.2**: Refuted (evidence contradicts)

## AGGREGATION
1. Score each assertion independently
2. Claim verdict = importance-weighted mean of FACT assertion scores
3. CONTEXT assertions are modifiers, not drivers
4. Global verified_score = importance-weighted mean of claim verdicts

## CONTENT_UNAVAILABLE Handling
If evidence has `content_status: "unavailable"`:
- Lower explainability_score (we couldn't read the source)
- Assertion stays AMBIGUOUS, NOT refuted
- This is NOT evidence against the claim

# OUTPUT FORMAT
```json
{{
  "claim_verdicts": [
    {{
      "claim_id": "c1",
      "verdict_score": 0.85,
      "status": "verified",
      "assertion_verdicts": [
        {{
          "assertion_key": "event.location.city",
          "dimension": "FACT",
          "score": 0.9,
          "status": "verified",
          "evidence_count": 2,
          "rationale": "Multiple sources confirm Miami"
        }},
        {{
          "assertion_key": "event.time_reference",
          "dimension": "CONTEXT",
          "score": 0.8,
          "status": "verified",
          "evidence_count": 1,
          "rationale": "Time zone context verified"
        }}
      ],
      "reason": "Location and timing confirmed by official sources."
    }}
  ],
  "verified_score": 0.85,
  "explainability_score": 0.9,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}
```

Write rationale and reason in **{lang_name}** ({lang}).
Return valid JSON.
"""


def build_score_evidence_structured_prompt(*, claims_data: list[dict], evidence_by_assertion: dict) -> str:
    return f"""Score these claims with per-assertion verdicts.

Claims with Assertions:
{json.dumps(claims_data, indent=2, ensure_ascii=False)}

Evidence by Assertion:
{json.dumps(evidence_by_assertion, indent=2, ensure_ascii=False)}

Remember: CONTEXT cannot refute FACT. Score each assertion independently.
Return JSON.
"""


def build_stance_matrix_instructions(*, num_sources: int, pass_type: str) -> str:
    normalized = (pass_type or "").strip().upper()
    if normalized not in STANCE_PASS_TYPES:
        normalized = STANCE_PASS_SINGLE

    pass_rules = ""
    if normalized == STANCE_PASS_SUPPORT_ONLY:
        pass_rules = """PASS MODE: SUPPORT_ONLY
- You MUST NOT output REFUTE or MIXED.
- Output SUPPORT only if a direct quote/span explicitly supports the claim/assertion.
- If support is unclear or only contextual, output CONTEXT (or IRRELEVANT if unrelated).
"""
    elif normalized == STANCE_PASS_REFUTE_ONLY:
        pass_rules = """PASS MODE: REFUTE_ONLY
- You MUST NOT output SUPPORT or MIXED.
- Output REFUTE only if a direct quote/span explicitly contradicts the claim/assertion.
- If contradiction is unclear or only contextual, output CONTEXT (or IRRELEVANT if unrelated).
"""
    else:
        pass_rules = """PASS MODE: SINGLE_PASS (attempt-to-refute)
- First, attempt to find explicit contradictions. If found, output REFUTE with the contradiction quote.
- If no contradiction exists, output SUPPORT only if a direct quote/span explicitly supports the claim.
- If neither applies, output CONTEXT (or IRRELEVANT if unrelated).
"""

    return f"""You are an Evidence Analyst.
Your task is to map each Search Source to its BEST matching Claim AND Assertion.

## CRITICAL CONTRACT
- You MUST output EXACTLY {num_sources} matrix rows, one for each source_index from 0 to {num_sources - 1}.
- NEVER return an empty matrix. If unsure about a source, output a row with stance="CONTEXT".
- Every row MUST include: source_index, claim_id (or null), stance, quote (string or null), and optional assertion_key.

## MATCHING SOURCES TO CLAIMS
- Each claim has a `search_query` field showing what search query was used to retrieve sources.
- If a source's content/title matches the topic of a claim's `search_query`, assign that `claim_id`.
- Sources were retrieved FOR specific claims - use this context.

{pass_rules}

## Relevance Scoring
- Assign `relevance` (0.0-1.0).
- If relevance < 0.4, you MUST mark stance as `IRRELEVANT` or `CONTEXT`.
- If content is [UNAVAILABLE], judge relevance based on title/snippet.

## Quote Requirements
- For `SUPPORT` or `REFUTE`: quote MUST be non-empty and directly relevant.
- For `CONTEXT`, `IRRELEVANT`, `MENTION`: quote can be null or empty string.

## Output JSON Schema
```json
{{
  "matrix": [
    {{
      "source_index": 0,
      "claim_id": "c1",
      "assertion_key": "event.location.city", // or null
      "stance": "SUPPORT",
      "relevance": 0.9,
      "quote": "Direct quote from text...",
      "reason": "Explain why..."
    }}
  ]
}}
```
"""


def build_stance_matrix_prompt(*, claims_lite: list[dict], sources_lite: list[dict]) -> str:
    return f"""Build the Evidence Matrix for these sources.

CLAIMS:
{json.dumps(claims_lite, indent=2)}

SOURCES:
{json.dumps(sources_lite, indent=2)}

Return the result in JSON format with key "matrix".
"""
