# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


# Static instruction blocks used for prefix caching matching.
# These blocks are large enough to exceed 1024 tokens when combined with skill-specific instructions.
# They serve as a "Methodological Appendix" for the LLM.

UNIVERSAL_METHODOLOGY_APPENDIX = """
### UNIVERSAL METHODOLOGY APPENDIX (STATIC)
1. **Universal Fact-Checking Standards**:
   - **Core Claim Definition**: An assertion of fact that can be objectively verified against reliable evidence. It is not an opinion, prediction, or subjective evaluation.
   - **Verifiability Criteria**: Specific (concrete details), Falsifiable (can be disproven), Sourceable (originates from a source).
   - **Claim Types**: Core (central thesis), Numeric (stats/money), Timeline (sequence), Attribution ("X said Y"), Sidefact (context).
   - **Evidence Requirements**: Primary Source (direct origin), Independent Verification (2+ non-affiliated sources), Quote Verification (exact match).
   - **Anti-Hallucination**: No inferring details not present. No paraphrasing that alters meaning. Ambiguity = capture exact wording.

2. **Verdict Scoring Standards**:
   - **Verified Score (0-1)**:
     - 0.0-0.2 (False): Direct contradiction by primary sources.
     - 0.2-0.4 (Unlikely/Misleading): True elements but deceptive context/omission.
     - 0.4-0.6 (Unproven): Conflicting evidence, no objective arbiter.
     - 0.6-0.8 (Plausible): Secondary sources support, but minor discrepancies or lack of independence.
     - 0.8-1.0 (True): Confirmed by multiple independent Tier 1 sources/visuals.
   - **Danger Score (0-1)**:
     - 0.0-0.2: Harmless news/opinion.
     - 0.2-0.5: Heated rhetoric/clickbait.
     - 0.5-0.8: Medical misinformation, hate speech dog-whistles.
     - 0.8-1.0: Incitement to violence, doxxing, illegal acts.
   - **Independence Principle**: Two sources citing similar text are NOT independent (circular reporting). True independence = distinct reporting chains.

3. **Stance Clustering & Search**:
   - **Stance**: Support (agrees with premise), Contradict (disagrees/debunks), Neutral (mentions without sidebarring).
   - **Relevance**: High (0.8-1.0, direct subject match), Medium (0.4-0.7, related topic), Low (0-0.3, barely related).
   - **Query Strategy**: Use Proper Nouns + Verbs. Use "2024/2025" for recency. Dual-stack: English (global) + Local (ground truth).

4. **JSON Output Requirement**:
   - **CRITICAL**: You must output a single valid JSON object.
   - Do not include markdown fencing (```json ... ```) if possible, but if you do, ensure the content inside is valid JSON.
   - No prologue or epilogue text. Just the JSON.
"""
