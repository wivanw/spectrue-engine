# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json

STANCE_PASS_SUPPORT_ONLY = "SUPPORT_ONLY"
STANCE_PASS_REFUTE_ONLY = "REFUTE_ONLY"
STANCE_PASS_SINGLE = "SINGLE_PASS"
STANCE_PASS_TYPES = {STANCE_PASS_SUPPORT_ONLY, STANCE_PASS_REFUTE_ONLY, STANCE_PASS_SINGLE}


# ==============================================================================
# SHARED SCORING CONSTANTS (Single Source of Truth)
# ==============================================================================

SCORING_SCALE = """# SCORING SCALE (-1.0 to 1.0)
- **0.8 - 1.0 (Verified)**: Strong confirmation (direct quotes, official consensus).
- **0.6 - 0.8 (Plausible)**: Supported, but may lack direct confirmation.
- **0.4 - 0.6 (Ambiguous)**: Conflicting evidence. (Use 0.5 only if evidence effectively cancels out).
- **0.2 - 0.4 (Unlikely)**: Evidence suggests the claim is doubtful.
- **0.0 - 0.2 (Refuted)**: Evidence contradicts the claim.
- **-1.0 (Unverified/Unknown)**: NO usable evidence found. Cannot judge at all. (Do NOT use 0.5 for lack of info)."""

RGBA_EXPLANATION = """# RGBA SCORING
Return `rgba`: [R, G, B, A] where:
- R = danger (0=harmless, 1=dangerous misinformation)
- G = verdict_score (same as verdict_score, use -1.0 if unknown)
- B = style (0=biased, 1=neutral)
- A = explainability (0=no evidence, 1=strong direct quotes)"""

VERDICT_VALUES = ["verified", "refuted", "ambiguous", "unverified", "partially_verified"]


def _language_contract(lang_name: str, lang: str) -> str:
    """Shared language/localization instructions."""
    return f"""# LANGUAGE CONTRACT
- Respond ONLY in **{lang_name}** ({lang}).
- Do NOT switch languages.
- Write `reason` and `rationale` ENTIRELY in **{lang_name}**."""


def _claim_verdict_example() -> str:
    """Shared JSON example for claim verdict with RGBA."""
    return """{{"claim_id": "c1", "verdict_score": 0.9, "verdict": "verified", "reason": "...", "rgba": [0.1, 0.9, 0.85, 0.8]}}"""


# ==============================================================================
# BATCH SCORING (Multiple Claims)
# ==============================================================================

def build_score_evidence_instructions(*, lang_name: str, lang: str) -> str:
    return f"""You are the Spectrue Verdict Engine.
Your task is to classify the reliability of claims based *strictly* on the provided Evidence.

# INPUT DATA
- **Claims**: Note the `importance` (0.0-1.0). High importance = Core Thesis.
- **Evidence**: Look for "📌 QUOTE" segments. `stance` is a hint; always verify against the quote/excerpt.
- **Metadata**: `source_reliability_hint` is context, not a hard rule.
- **Consistency Rule**: If a claim has `matched_evidence_count > 0`, you MUST NOT say "no sources/evidence". Say the evidence is indirect, insufficient, or lacks direct confirmation.

{_language_contract(lang_name, lang)}
- Do NOT mention other claims, other claim IDs, or "c1/c2" comparisons in user-facing text.

{SCORING_SCALE}

# AGGREGATION LOGIC (Global Score)
Do NOT compute a global `verified_score`. The engine computes it deterministically in code.
Set `verified_score` to **0.5** as a placeholder.

# WRITING GUIDELINES (User-Facing Text Only)
- **Tone**: Natural, journalistic style.
- **FORBIDDEN TERMS**: Do not use "JSON", "dataset", "primary source", "relevance score", "cap" in readable text.
  - Instead of "lack of primary source", say "no official confirmation found".

# OUTPUT FORMAT
Return valid JSON:
{{
  "claim_verdicts": [
    {_claim_verdict_example()}
  ],
  "verified_score": 0.5,
  "explainability_score": 0.8,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}

# PER-CLAIM RGBA
Each claim_verdict MUST include `rgba`: [R=danger, G=verdict_score, B=style, A=explainability].
Values 0-1 (except G which can be -1.0). Each claim may have DIFFERENT R, B, A based on its content and evidence.

# GLOBAL SCORES EXPLANATION
- **verified_score**: Placeholder only (set to 0.5). Engine computes the true global score.
- **explainability_score** (0.0-1.0): How well evidence supports rationale. 1.0 = backed by direct quotes.
- **danger_score** (0.0-1.0): How harmful if acted upon? (0.0 = harmless, 1.0 = dangerous)
- **style_score** (0.0-1.0): How neutral is writing style? (0.0 = biased, 1.0 = neutral)
"""


# ==============================================================================
# SINGLE CLAIM SCORING (Parallel Mode)
# ==============================================================================

def build_single_claim_scoring_instructions(*, lang_name: str, lang: str) -> str:
    """Instructions for scoring a SINGLE claim (parallel scoring mode)."""
    return f"""You are the Spectrue Verdict Engine.
Score this SINGLE claim based strictly on the provided Evidence.

{SCORING_SCALE}

# CRITICAL: NO EVIDENCE HANDLING
If the provided evidence is empty, unrelated, or insufficient to form ANY opinion:
- Set `verdict_score` to **-1.0**.
- Set `verdict` to "unverified".
- Explain in `reason` that evidence is missing.
- **DO NOT** use 0.5 (Ambiguous) for missing evidence. 0.5 is for conflicting evidence.

{RGBA_EXPLANATION}

# OUTPUT FORMAT
Return JSON:
{{
  "claim_id": "c1",
  "verdict_score": -1.0,
  "verdict": "unverified",
  "reason": "Explanation in {lang_name}...",
  "rgba": [0.1, -1.0, 0.85, 0.0]
}}

{_language_contract(lang_name, lang)}
Be concise.
"""


def build_single_claim_scoring_prompt(*, claim_info: dict, evidence: list[dict]) -> str:
    """Prompt for scoring a SINGLE claim."""
    return f"""Score this claim based on the evidence.

Claim:
{json.dumps(claim_info, indent=2, ensure_ascii=False)}

Evidence:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Return JSON.
"""


# ==============================================================================
# EVIDENCE PROMPTS
# ==============================================================================

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


# ==============================================================================
# STRUCTURED SCORING (Per-Assertion)
# ==============================================================================

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

{SCORING_SCALE}

## AGGREGATION
1. Score each assertion independently
2. Claim verdict = importance-weighted mean of FACT assertion scores
3. CONTEXT assertions are modifiers, not drivers
4. Do NOT aggregate global verified_score (set placeholder 0.5)

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
  "verified_score": 0.5,
  "explainability_score": 0.9,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}
```

{_language_contract(lang_name, lang)}
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
        pass_rules = """PASS MODE: SINGLE_PASS (balanced)
- SUPPORT if the source reports the same facts/discovery as the claim, even if paraphrased or summarized.
- SUPPORT if the source is about the same scientific topic and confirms the key details.
- News articles covering the same research paper or discovery = SUPPORT (not CONTEXT).
- Output REFUTE only if a direct quote explicitly CONTRADICTS the claim.
- IRRELEVANT if source is about a completely different topic.
- CONTEXT only for tangentially related content that neither confirms nor denies.
- When in doubt between SUPPORT and CONTEXT for related content, prefer SUPPORT.
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
{
  "matrix": [
    {
      "source_index": 0,
      "claim_id": "c1",
      "assertion_key": "event.location.city", // or null 
      "stance": "SUPPORT",
      "relevance": 0.9,
      "quote": "Direct quote from text...",
      "evidence_role": "direct", // direct | indirect | mention_only
      "covers": ["entity","time","location","quantity","attribution","causal","other"], // components covered
      "reason": "Explain why..."
    }
  ]
}
```
"""


def build_stance_matrix_prompt(*, claims_lite: list[dict], sources_lite: list[dict]) -> str:
    """Build prompt body for Evidence Matrix (stance clustering).

    Contract:
    - Pure formatting: no logic, no filtering.
    - Caller provides claims_lite and sources_lite (already sanitized/capped).
    """
    return (
        "Map each source to the best matching claim and extract a direct quote when possible.\n\n"
        "Claims:\n"
        f"{json.dumps(claims_lite or [], indent=2, ensure_ascii=False)}\n\n"
        "Sources:\n"
        f"{json.dumps(sources_lite or [], indent=2, ensure_ascii=False)}\n\n"
        "Return JSON matching the requested schema."
    )
