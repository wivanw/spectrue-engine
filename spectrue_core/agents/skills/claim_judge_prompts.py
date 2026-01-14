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


def _format_stats_section(frame: ClaimFrame, *, include_v2: bool = False) -> str:
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
    if include_v2:
        lines.extend(
            [
                f"Exact duplicates (urls): {stats.exact_dupes_total}",
                f"Similar clusters: {stats.similar_clusters_total}",
                f"Unique publishers: {stats.publishers_total}",
                f"Support precision publishers: {stats.support.precision_publishers}",
                f"Support corroboration clusters: {stats.support.corroboration_clusters}",
                f"Refute precision publishers: {stats.refute.precision_publishers}",
                f"Refute corroboration clusters: {stats.refute.corroboration_clusters}",
                (
                    "Confirmation counts: "
                    f"C_precise={frame.confirmation_counts.C_precise}, "
                    f"C_corr={frame.confirmation_counts.C_corr}, "
                    f"C_total={frame.confirmation_counts.C_total}"
                ),
            ]
        )

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
    *,
    ui_locale: str = "en",
    analysis_mode: str = "deep",
) -> str:
    """
    Build prompt for claim judging.
    
    The judge produces RGBA scores and verdict for a single claim.
    Output must be returned unchanged to the frontend.
    
    Args:
        frame: ClaimFrame with claim and evidence
        evidence_summary: Optional pre-analyzed evidence summary
        ui_locale: UI language for explanation output (e.g., "uk", "en")
    
    Returns:
        Formatted prompt string
    """
    evidence_section = _format_evidence_for_judge(frame.evidence_items)
    summary_section = _format_evidence_summary(evidence_summary)
    stats_section = _format_stats_section(frame, include_v2=(analysis_mode == "deep_v2"))

    # Get URLs for sources_used constraint
    available_urls = [item.url for item in frame.evidence_items]
    urls_list = "\n".join(f"  - {url}" for url in available_urls) if available_urls else "  (none)"

    # Use UI locale for prompt language (not claim language)
    # This ensures explanations are in the user's interface language
    lang = ui_locale.lower()[:2]
    from spectrue_core.agents.prompts import get_prompt
    
    # Try to get specific locale prompt, fallback to English if missing
    prompt_template = get_prompt(lang, "prompts.claim_judge")
    if not prompt_template or "Prompt key" in prompt_template:
        prompt_template = get_prompt("en", "prompts.claim_judge")

    # If even English is missing (should not happen if files are correct), use fallback code
    if not prompt_template or "Prompt key" in prompt_template:
        return _fallback_english_prompt(frame, evidence_section, summary_section, stats_section, urls_list, ui_locale)

    # Fill template variables
    # We must ensure keys match what's in the YAML files
    try:
        prompt = prompt_template.format(
            claim_id=frame.claim_id,
            claim_text=frame.claim_text,
            claim_language=frame.claim_language,
            ui_locale=ui_locale,  # Add UI locale for explicit instruction
            context_text=frame.context_excerpt.text,
            evidence_section=evidence_section,
            summary_section=summary_section,
            stats_section=stats_section,
            urls_list=urls_list
        )
    except KeyError:
        # Fallback if template has broken keys
        return _fallback_english_prompt(frame, evidence_section, summary_section, stats_section, urls_list, ui_locale)

    return prompt


def _fallback_english_prompt(frame, evidence_section, summary_section, stats_section, urls_list, ui_locale: str = "en"):
    """Hardcoded English fallback just in case."""
    return f"""You are a fact-checking judge. Evaluate the following claim based on the provided evidence.

## CLAIM
ID: {frame.claim_id}
Text: "{frame.claim_text}"
Claim Language: {frame.claim_language}

## EVIDENCE
{evidence_section}

## SUMMARY
{summary_section}

## EVIDENCE STATS
{stats_section}

## OUTPUT FORMAT
Return JSON with: claim_id, rgba, confidence, verdict, explanation, sources_used, missing_evidence.
Explanation MUST be in {ui_locale} (user's interface language).
"""




def build_claim_judge_system_prompt(*, lang: str = "en") -> str:
    """Build system prompt for claim judge."""
    # Try to load localized system prompt
    from spectrue_core.agents.prompts import get_prompt
    
    system_prompt = get_prompt(lang[:2], "prompts.claim_judge_system")
    if system_prompt and "Prompt key" not in system_prompt:
        return system_prompt

    # Fallback to English hardcoded
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
