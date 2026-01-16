# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Prompt builders for RGBA audit tasks."""

from __future__ import annotations

from spectrue_core.schema.claim_frame import ClaimFrame, EvidenceItemFrame


def build_claim_audit_system_prompt() -> str:
    return (
        "You are an audit annotator. Output JSON only that matches the schema.\n"
        "Do NOT provide scores, verdicts, or commentary outside JSON.\n"
        "Avoid markdown, code fences, or prose. JSON only.\n\n"
        "BAD (do not do this):\n"
        "The claim seems plausible because...\n\n"
        "GOOD (example shape):\n"
        "{\n"
        "  \"claim_id\": \"c1\",\n"
        "  \"predicate_type\": \"event\",\n"
        "  \"truth_conditions\": [\"Condition A\"],\n"
        "  \"expected_evidence_types\": [\"official_statement\"],\n"
        "  \"failure_modes\": [\"no_sources\"],\n"
        "  \"assertion_strength\": \"strong\",\n"
        "  \"risk_facets\": [\"public_safety\"],\n"
        "  \"honesty_facets\": [\"selective_context\"],\n"
        "  \"what_would_change_mind\": [\"Official report contradicting the claim\"],\n"
        "  \"audit_confidence\": 0.75\n"
        "}"
    )


def build_claim_audit_prompt(frame: ClaimFrame) -> str:
    excerpt = frame.context_excerpt.text.strip()
    stats = frame.evidence_stats
    return (
        "Audit the claim. Provide structured audit fields only.\n\n"
        f"Claim ID: {frame.claim_id}\n"
        f"Claim Text: {frame.claim_text}\n"
        f"Language: {frame.claim_language}\n"
        f"Context Excerpt: {excerpt}\n"
        "Evidence Stats:\n"
        f"  total_sources={stats.total_sources}\n"
        f"  support_sources={stats.support_sources}\n"
        f"  refute_sources={stats.refute_sources}\n"
        f"  context_sources={stats.context_sources}\n"
    )


def build_evidence_audit_system_prompt() -> str:
    return (
        "You are an evidence audit annotator. Output JSON only that matches the schema.\n"
        "Do NOT produce final scores or narrative summaries.\n"
        "Avoid markdown, code fences, or commentary. JSON only.\n"
        "Return the exact Source ID provided in the prompt.\n\n"
        "BAD (do not do this):\n"
        "This source supports the claim strongly because...\n\n"
        "GOOD (example shape):\n"
        "{\n"
        "  \"claim_id\": \"c1\",\n"
        "  \"evidence_id\": \"e1\",\n"
        "  \"source_id\": \"s1\",\n"
        "  \"stance\": \"support\",\n"
        "  \"directness\": \"direct\",\n"
        "  \"specificity\": \"high\",\n"
        "  \"quote_integrity\": \"ok\",\n"
        "  \"extraction_confidence\": 0.9,\n"
        "  \"novelty_vs_copy\": \"original\",\n"
        "  \"dependency_hints\": [],\n"
        "  \"audit_confidence\": 0.8\n"
        "}"
    )


def build_evidence_audit_prompt(frame: ClaimFrame, evidence: EvidenceItemFrame) -> str:
    return (
        "Audit the evidence item against the claim. Provide structured audit fields only.\n\n"
        f"Claim ID: {frame.claim_id}\n"
        f"Claim Text: {frame.claim_text}\n"
        f"Evidence ID: {evidence.evidence_id}\n"
        f"Source ID: {evidence.source_id}\n"
        f"URL: {evidence.url}\n"
        f"Title: {evidence.title or ''}\n"
        f"Snippet: {evidence.snippet or ''}\n"
        f"Quote: {evidence.quote or ''}\n"
        f"Trust Tier: {evidence.source_tier or ''}\n"
    )
