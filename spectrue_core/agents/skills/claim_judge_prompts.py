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

## JUDGMENT INSTRUCTIONS

Produce an RGBA verdict for this claim:

**R (Danger: 0.0-1.0)**: How harmful is it if this claim is believed?
  - 0.0 = Not harmful at all
  - 1.0 = Extremely dangerous (medical misinformation, incitement, etc.)

**G (Veracity: 0.0-1.0)**: How factually accurate is this claim?
  - 0.0 = Completely false
  - 0.5 = Unverifiable or mixed
  - 1.0 = Fully supported by evidence

**B (Honesty: 0.0-1.0)**: Is the claim presented in good faith?
  - 0.0 = Deliberately misleading
  - 0.5 = Unclear intent
  - 1.0 = Honest presentation

**A (Explainability: 0.0-1.0)**: How well can we trace and explain the verdict?
  - 0.0 = No traceable evidence
  - 1.0 = Clear, quotable evidence trail

**Verdict**: Choose one of:
  - "Supported" - Evidence confirms the claim
  - "Refuted" - Evidence contradicts the claim
  - "NEI" - Not Enough Information to decide
  - "Mixed" - Evidence is contradictory
  - "Unverifiable" - Claim cannot be verified with available sources

**Confidence**: Your overall confidence in this verdict (0.0-1.0)

**Explanation**: Brief explanation of your verdict (1-3 sentences)

**sources_used**: ONLY include URLs from this list that you actually used:
{urls_list}

**missing_evidence**: What additional evidence would strengthen or change the verdict?

## OUTPUT FORMAT

Respond with valid JSON:
{{
  "claim_id": "{frame.claim_id}",
  "rgba": {{
    "R": 0.0,
    "G": 0.0,
    "B": 0.0,
    "A": 0.0
  }},
  "confidence": 0.0,
  "verdict": "...",
  "explanation": "...",
  "sources_used": ["url1", "url2"],
  "missing_evidence": ["...", "..."]
}}

Judge this claim fairly and objectively."""

    return prompt


def build_claim_judge_system_prompt() -> str:
    """Build system prompt for claim judge."""
    return """You are an impartial fact-checking judge. Your role is to evaluate claims based on evidence.

Key principles:
1. Base verdicts ONLY on provided evidence, not prior knowledge
2. Be conservative - if uncertain, lean toward NEI
3. High-trust sources (Tier A, B) carry more weight than low-trust ones
4. Direct quotes are stronger evidence than summaries
5. Conflicting evidence should result in "Mixed" or lower confidence
6. sources_used must ONLY contain URLs that appear in the evidence list
7. Explain your reasoning clearly and concisely

RGBA scoring guidance:
- R (Danger): Consider health, safety, financial, or social harm
- G (Veracity): Focus on factual accuracy, not opinions
- B (Honesty): Consider context, framing, and potential intent
- A (Explainability): How well can you cite evidence for your verdict?"""