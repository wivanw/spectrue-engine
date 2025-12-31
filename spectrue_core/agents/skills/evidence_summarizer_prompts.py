# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Evidence Summarizer prompt builder for per-claim deep analysis.

Builds prompts for the evidence summarizer skill that categorizes
evidence items by stance and identifies gaps.
"""

from __future__ import annotations

from spectrue_core.schema.claim_frame import ClaimFrame, EvidenceItemFrame


def _format_evidence_item(item: EvidenceItemFrame, index: int) -> str:
    """Format a single evidence item for the prompt."""
    lines = [f"[{item.evidence_id}] Source {index + 1}:"]
    lines.append(f"  URL: {item.url}")
    
    if item.title:
        lines.append(f"  Title: {item.title}")
    
    if item.source_tier:
        lines.append(f"  Trust Tier: {item.source_tier}")
    
    if item.quote:
        quote_preview = item.quote[:300] + "..." if len(item.quote) > 300 else item.quote
        lines.append(f"  Quote: \"{quote_preview}\"")
    elif item.snippet:
        snippet_preview = item.snippet[:300] + "..." if len(item.snippet) > 300 else item.snippet
        lines.append(f"  Snippet: {snippet_preview}")
    
    if item.stance:
        lines.append(f"  Initial Stance: {item.stance}")
    
    if item.relevance is not None:
        lines.append(f"  Relevance: {item.relevance:.2f}")
    
    return "\n".join(lines)


def build_evidence_summarizer_prompt(frame: ClaimFrame) -> str:
    """
    Build prompt for evidence summarization.
    
    Args:
        frame: ClaimFrame containing claim and evidence
    
    Returns:
        Formatted prompt string
    """
    evidence_section = ""
    if frame.evidence_items:
        items = [
            _format_evidence_item(item, i) 
            for i, item in enumerate(frame.evidence_items)
        ]
        evidence_section = "\n\n".join(items)
    else:
        evidence_section = "No evidence items available."
    
    prompt = f"""You are an evidence analyst. Your task is to categorize the provided evidence for a specific claim.

## CLAIM TO ANALYZE

Claim ID: {frame.claim_id}
Claim Text: "{frame.claim_text}"
Claim Language: {frame.claim_language}

## CONTEXT EXCERPT

{frame.context_excerpt.text}

## EVIDENCE ITEMS

{evidence_section}

## YOUR TASK

Analyze each evidence item and categorize it:

1. **supporting_evidence**: Evidence that confirms or supports the claim
2. **refuting_evidence**: Evidence that contradicts or refutes the claim
3. **contextual_evidence**: Evidence that provides background but doesn't directly support/refute

For each categorized item, provide:
- The evidence_id (from brackets like [abc123])
- A brief reason explaining why it fits that category

Also identify:
- **evidence_gaps**: What types of evidence are missing that would help reach a confident verdict?
- **conflicts_present**: Are there contradictions between sources?

## OUTPUT FORMAT

Respond with valid JSON matching this structure:
{{
  "supporting_evidence": [
    {{"evidence_id": "...", "reason": "..."}}
  ],
  "refuting_evidence": [
    {{"evidence_id": "...", "reason": "..."}}
  ],
  "contextual_evidence": [
    {{"evidence_id": "...", "reason": "..."}}
  ],
  "evidence_gaps": ["...", "..."],
  "conflicts_present": true/false
}}

Analyze carefully and categorize each evidence item."""

    return prompt


def build_evidence_summarizer_system_prompt() -> str:
    """Build system prompt for evidence summarizer."""
    return """You are a precise evidence analyst. You categorize evidence by its relationship to claims.

Rules:
- Use exact evidence_id values from the input
- Be objective in categorization
- SUPPORT means the evidence directly backs the claim
- REFUTE means the evidence directly contradicts the claim
- CONTEXT means the evidence provides background without taking a stance
- Identify missing evidence types that would strengthen analysis
- Flag conflicts when sources disagree"""
