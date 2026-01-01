# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json


VALID_STANCES = {"SUPPORT", "REFUTE", "MIXED", "CONTEXT", "NEUTRAL"}


def build_evidence_matrix_instructions(*, num_sources: int) -> str:
    return f"""You are an Evidence Analyst. 
Your task is to map each Search Source to its BEST matching Claim AND Assertion.

## CRITICAL CONTRACT
- You MUST output EXACTLY {num_sources} matrix rows, one for each source_index from 0 to {num_sources - 1}.
- NEVER return an empty matrix. If unsure about a source, output a row with stance="CONTEXT".
- Every row MUST include: source_index, claim_id (or null), stance, quote (string or null), and optional assertion_key.

## Methodology
1. **1:1 Mapping**: Each Source must be mapped to EXACTLY ONE Claim (the most relevant one).
   - If a source supports/refutes a specific ASSERTION (e.g. location, time, amount), map it to that `assertion_key`.
   - If it covers the whole claim generally, leave `assertion_key` null.
2. **Stance Classification**:
   - `SUPPORT`: Source confirms the claim/assertion is TRUE.
   - `REFUTE`: Source proves the claim/assertion is FALSE.
   - `MIXED`: Source says it's complicated / partially true.
   - `MENTION`: Topic is mentioned but no clear verdict.
   - `CONTEXT`: Source is background/tangentially related but not evidence.
   - `IRRELEVANT`: Source is completely unrelated to any claim.
3. **Relevance Scoring**: Assign `relevance` (0.0-1.0).
   - If relevance < 0.4, you MUST mark stance as `IRRELEVANT` or `CONTEXT`.
   - If content is [UNAVAILABLE], judge relevance based on title/snippet. Do NOT penalize relevance just because content is missing if the source seems authoritative.
4. **Quote Extraction**:
   - For `SUPPORT`, `REFUTE`, `MIXED`: quote MUST be non-empty and directly relevant.
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
```"""


def build_evidence_matrix_prompt(*, claims_lite: list[dict], sources_lite: list[dict]) -> str:
    return f"""Build the Evidence Matrix for these sources.

CLAIMS:
{json.dumps(claims_lite, indent=2)}

SOURCES:
{json.dumps(sources_lite, indent=2)}

Return the result in JSON format with key "matrix".
"""
