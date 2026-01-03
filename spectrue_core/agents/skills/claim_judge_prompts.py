# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Claim Judge prompt builder for per-claim deep analysis.

Builds prompts for the claim judge skill that produces RGBA verdicts
for individual claims based on their evidence.
"""

from __future__ import annotations

from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    EvidenceItemFrame,
    EvidenceSummary,
)


def _format_evidence_for_judge(items: tuple[EvidenceItemFrame, ...]) -> str:
    """Format evidence items for judge prompt."""
    if not items:
        return "No evidence available."

    lines: list[str] = []
    for i, item in enumerate(items):
        entry = [f"[{item.evidence_id}] Source {i + 1}:"]
        entry.append(f"  URL: {item.url}")

        if item.title:
            entry.append(f"  Title: {item.title}")

        if item.source_tier:
            entry.append(f"  Trust Tier: {item.source_tier}")

        if item.stance:
            entry.append(f"  Stance: {item.stance}")

        if item.quote:
            quote_preview = item.quote[:400] + "..." if len(item.quote) > 400 else item.quote
            entry.append(f"  Quote: \"{quote_preview}\"")
        elif item.snippet:
            snippet_preview = item.snippet[:400] + "..." if len(item.snippet) > 400 else item.snippet
            entry.append(f"  Content: {snippet_preview}")

        lines.append("\n".join(entry))

    return "\n\n".join(lines)


def _format_evidence_summary(summary: EvidenceSummary | None) -> str:
    """Format evidence summary for judge prompt."""
    if summary is None:
        return "No pre-analyzed summary available."

    lines: list[str] = []

    if summary.supporting_evidence:
        lines.append("Supporting evidence:")
        for ref in summary.supporting_evidence:
            lines.append(f"  - [{ref.evidence_id}]: {ref.reason}")

    if summary.refuting_evidence:
        lines.append("Refuting evidence:")
        for ref in summary.refuting_evidence:
            lines.append(f"  - [{ref.evidence_id}]: {ref.reason}")

    if summary.contextual_evidence:
        lines.append("Contextual evidence:")
        for ref in summary.contextual_evidence:
            lines.append(f"  - [{ref.evidence_id}]: {ref.reason}")

    if summary.evidence_gaps:
        lines.append(f"Evidence gaps: {', '.join(summary.evidence_gaps)}")

    if summary.conflicts_present:
        lines.append("⚠️ Conflicting evidence detected")

    return "\n".join(lines) if lines else "Summary is empty."


def _format_stats_section(frame: ClaimFrame) -> str:
    """Format evidence stats for judge prompt."""
    stats = frame.evidence_stats
    lines = [
        f"Total sources: {stats.total_sources}",
        f"Supporting sources: {stats.support_sources}",
        f"Refuting sources: {stats.refute_sources}",
        f"Context sources: {stats.context_sources}",
        f"High-trust sources: {stats.high_trust_sources}",
        f"Direct quotes: {stats.direct_quotes}",
    ]

    if stats.conflicting_evidence:
        lines.append("⚠️ Evidence contains contradictions")

    if stats.missing_sources:
        lines.append("⚠️ No sources found")
    elif stats.missing_direct_quotes:
        lines.append("ℹ️ No direct quotes available")

    return "\n".join(lines)


def build_claim_judge_prompt(
    frame: ClaimFrame,
    evidence_summary: EvidenceSummary | None = None,
) -> str:
    """
    Build prompt for claim judging.
    
    The judge produces RGBA scores and verdict for a single claim.
    Output must be returned unchanged to the frontend.
    
    Args:
        frame: ClaimFrame with claim and evidence
        evidence_summary: Optional pre-analyzed evidence summary
    
    Returns:
        Formatted prompt string
    """
    evidence_section = _format_evidence_for_judge(frame.evidence_items)
    summary_section = _format_evidence_summary(evidence_summary)
    stats_section = _format_stats_section(frame)

    # Get URLs for sources_used constraint
    available_urls = [item.url for item in frame.evidence_items]
    urls_list = "\n".join(f"  - {url}" for url in available_urls) if available_urls else "  (none)"

    prompt = f"""You are a fact-checking judge. Evaluate the following claim based on the provided evidence and produce a verdict.

## CLAIM TO JUDGE

Claim ID: {frame.claim_id}
Claim Text: "{frame.claim_text}"
Claim Language: {frame.claim_language}

## ORIGINAL CONTEXT

{frame.context_excerpt.text}

## EVIDENCE ITEMS

{evidence_section}

## EVIDENCE SUMMARY (Pre-analyzed)

{summary_section}

## EVIDENCE STATISTICS

{stats_section}

## JUDGMENT METHODOLOGY

Follow this evaluation process:

1. **Evidence Assessment**: What do sources actually say? Name specific sources/domains.
2. **Manipulation Check**: Look for cherry-picking, missing context, sensationalism, loaded language.
3. **Gaps Identification**: What evidence is missing? Be explicit about limitations.
4. **Verdict**: Base ONLY on evidence. If evidence is insufficient, say so — use -1 for scores.

## RGBA SCORING GUIDANCE

**R (Danger: 0.0-1.0 or -1)**: Harm potential if this claim is believed
  - 0.0-0.2 = Harmless, informational only
  - 0.2-0.5 = Potentially misleading but low risk
  - 0.5-0.8 = Medical, financial, or safety misinformation
  - 0.8-1.0 = Dangerous incitement, doxxing, illegal advice
  - **-1** = Cannot assess (use when claim nature is unclear)

**G (Veracity: 0.0-1.0 or -1)**: Factual accuracy — MUST match verdict!
  - 0.0-0.2 = **Strong refutation** by authoritative sources
  - 0.2-0.4 = Mostly false, misleading with some true elements
  - 0.4-0.6 = **Insufficient evidence** / conflicting sources
  - 0.6-0.8 = Mostly supported by reliable sources
  - 0.8-1.0 = **Strong confirmation** by multiple independent authoritative sources
  - **-1** = Cannot determine (for NEI/Unverifiable verdicts ONLY)

**B (Honesty: 0.0-1.0 or -1)**: Presentation quality and good faith
  - 0.0-0.3 = Deliberately misleading, propaganda techniques
  - 0.3-0.5 = Sensationalism, loaded language, missing crucial context
  - 0.5-0.7 = Cherry-picking, absolutist claims ("always/never")
  - 0.7-0.9 = Minor framing issues but generally fair
  - 0.9-1.0 = Neutral, balanced presentation
  - **-1** = Cannot assess (claim too short or context unavailable)

**A (Explainability: 0.0-1.0 or -1)**: Evidence traceability
  - 0.0-0.2 = No traceable evidence found
  - 0.2-0.4 = Indirect evidence, no direct quotes
  - 0.4-0.6 = Some supporting snippets but no exact quotes
  - 0.6-0.8 = Direct quotes supporting verdict
  - 0.8-1.0 = Multiple independent quotes, strong evidence trail
  - **-1** = No relevant evidence at all

**Verdict**: Choose based on G (Veracity) score:
  - **"Supported"** → G should be 0.7-1.0
  - **"Refuted"** → G should be 0.0-0.3
  - **"Mixed"** → G should be 0.3-0.7 (conflicting evidence)
  - **"NEI"** → G = -1 (Not Enough Information)
  - **"Unverifiable"** → G = -1 (claim cannot be verified)

**Confidence**: Your overall confidence (0.0-1.0)
  - Lower if sources are weak, single, or contradictory
  - Lower if key context is missing

**Explanation**: STRUCTURED format, same language as claim ({frame.claim_language}):
  - Line 1: "Evidence:" - cite specific sources/domains
  - Line 2: "Gaps:" - what's missing (if any)
  - Line 3: "Verdict:" - conclude briefly
  - Optional Line 4: "Style:" - ONLY if B < 0.7 (manipulation detected)

**sources_used**: ONLY URLs from this list that you actually referenced:
{urls_list}

**missing_evidence**: What would strengthen or change the verdict?

## OUTPUT FORMAT

{{
  "claim_id": "{frame.claim_id}",
  "rgba": {{
    "R": <danger_score>,
    "G": <veracity_score_or_-1>,
    "B": <honesty_score>,
    "A": <explainability_score_or_-1>
  }},
  "confidence": <0.0_to_1.0>,
  "verdict": "<Supported|Refuted|NEI|Mixed|Unverifiable>",
  "explanation": "<structured_explanation_in_claim_language>",
  "sources_used": ["<urls_from_evidence>"],
  "missing_evidence": ["<what_is_missing>"]
}}

## CRITICAL RULES

1. **G MUST match verdict**: Supported→0.7-1.0, Refuted→0.0-0.3, NEI/Unverifiable→-1
2. **Use -1 honestly** when you cannot assess — do NOT guess with 0.5
3. **Never all zeros** — use -1 for insufficient evidence
4. **sources_used MUST be from evidence list** — don't invent URLs
5. **Explanation in claim language** — match {frame.claim_language} exactly

Judge this claim fairly and objectively."""

    return prompt


def build_claim_judge_system_prompt() -> str:
    """Build system prompt for claim judge."""
    return """You are an impartial fact-checking judge. Your role is to evaluate claims based on evidence.

## CORE PRINCIPLES

1. **Evidence-Based Only**: Verdicts based ONLY on provided evidence, not prior knowledge
2. **Conservative Approach**: If uncertain, lean toward NEI with G = -1
3. **Source Hierarchy**: High-trust sources (Tier A, B) > Low-trust (Tier C, D)
4. **Quote Priority**: Direct quotes > Summaries > Snippets
5. **Conflict Detection**: Contradictory evidence → "Mixed" verdict with 0.3-0.7 G
6. **Honest Uncertainty**: Use -1 when you genuinely cannot assess

## MANIPULATION DETECTION (for B - Honesty score)

Lower B score when you detect:
- **Cherry-picking**: Using isolated data points out of context
- **Absolutist language**: "always", "never", "all", "none" without evidence
- **Loaded/emotional language**: Fear-mongering, ad hominem attacks
- **Missing crucial context**: Key qualifications omitted
- **Sensationalism**: Exaggerated claims, clickbait framing
- **Conspiracy markers**: "they don't want you to know", "wake up"

## DANGER ASSESSMENT (for R - Danger score)

Increase R for claims that:
- Give medical advice without professional context
- Provide financial/investment recommendations
- Suggest dangerous actions or substances
- Incite violence or discrimination
- Spread health misinformation (vaccines, treatments)

## EXPLANATION FORMAT

Use structured labels:
- "Evidence: [cite specific sources/domains]"
- "Gaps: [what's missing]"
- "Verdict: [conclusion]"
- "Style: [manipulation issues]" — ONLY if B < 0.7

## ABSOLUTE RULES

1. **G MUST match verdict**: Supported→0.7-1.0, Refuted→0.0-0.3, Mixed→0.3-0.7, NEI/Unverifiable→-1
2. **sources_used MUST be from evidence list** — never invent URLs
3. **Explanation in claim's language** — match exactly
4. **Never all zeros** — use -1 for "cannot assess"
5. **-1 over 0.5** — honest uncertainty is better than fake neutrality"""
