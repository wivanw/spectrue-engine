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

"""
M126: Search Escalation Policy

Evidence-driven, deterministic escalation for Tavily retrieval.
Start cheap, expand only when observable quality signals indicate need.

Key components:
- QueryVariant: Deterministic query variants from claim fields (Q1/Q2/Q3)
- EscalationPass: Multi-pass search with escalating parameters (A→B→C→D)
- RetrievalOutcome: Observable quality signals for stop decisions
- Topic selection from structured claim fields (no text heuristics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EscalationConfig:
    """Configuration for escalation thresholds."""

    min_relevance_threshold: float = 0.2
    """Minimum relevance score to consider evidence usable."""

    min_usable_snippets: int = 2
    """Minimum snippets needed for early stop (combined with relevance)."""

    max_query_length: int = 80
    """Maximum length for query variants."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_relevance_threshold": self.min_relevance_threshold,
            "min_usable_snippets": self.min_usable_snippets,
            "max_query_length": self.max_query_length,
        }


DEFAULT_ESCALATION_CONFIG = EscalationConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Query Variants
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryVariant:
    """A deterministic query variant built from claim fields."""

    query_id: str
    """Query identifier: 'Q1', 'Q2', 'Q3'."""

    text: str
    """The query text (keyword-like, space-separated)."""

    strategy: str
    """Strategy used: 'anchor_tight', 'anchor_medium', 'broad'."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "strategy": self.strategy,
        }


def _extract_top_entities(claim: dict[str, Any], max_count: int = 3) -> list[str]:
    """Extract top subject entities from claim."""
    entities = claim.get("subject_entities", [])
    if not isinstance(entities, list):
        return []
    valid = [e for e in entities if isinstance(e, str) and len(e) >= 2]
    return valid[:max_count]


def _extract_seed_terms(claim: dict[str, Any], max_count: int = 6) -> list[str]:
    """Extract retrieval seed terms from claim."""
    terms = claim.get("retrieval_seed_terms", [])
    if not isinstance(terms, list):
        return []
    valid = [t for t in terms if isinstance(t, str) and len(t) >= 2]
    return valid[:max_count]


def _extract_date_anchor(claim: dict[str, Any]) -> str | None:
    """Extract explicit date from time_anchor if available."""
    time_anchor = claim.get("time_anchor")
    if not isinstance(time_anchor, dict):
        return None

    anchor_type = time_anchor.get("type", "unknown")
    if anchor_type not in ("explicit_date", "range"):
        return None

    # Try to get the date value
    date_val = time_anchor.get("value") or time_anchor.get("start")
    if isinstance(date_val, str) and len(date_val) >= 4:
        # Extract year or date portion (trim to reasonable length)
        return date_val[:10]  # e.g., "2024-01-15" or "2024"
    return None


def _truncate_query(query: str, max_length: int) -> str:
    """Truncate query to max_length, respecting word boundaries."""
    if len(query) <= max_length:
        return query
    truncated = query[:max_length]
    # Cut at last space to avoid partial words
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space].strip()
    return truncated.strip()


def build_query_variants(
    claim: dict[str, Any],
    config: EscalationConfig | None = None,
) -> list[QueryVariant]:
    """
    Build up to 3 query variants from claim structured fields.

    Q1 (anchor-tight): <top entities> <top seed terms> <date/range if explicit>
    Q2 (anchor-medium): <top entities> <top seed terms>
    Q3 (broad): <top entities> <2-3 seed terms>

    Each query <= max_query_length chars, keyword-like (no sentences).
    """
    cfg = config or DEFAULT_ESCALATION_CONFIG
    max_len = cfg.max_query_length

    entities = _extract_top_entities(claim, max_count=3)
    seed_terms = _extract_seed_terms(claim, max_count=6)
    date_anchor = _extract_date_anchor(claim)

    variants: list[QueryVariant] = []

    # Q1: anchor-tight (entities + seed terms + date)
    q1_parts = entities + seed_terms[:4]
    if date_anchor:
        q1_parts.append(date_anchor)
    q1_text = _truncate_query(" ".join(q1_parts), max_len)
    if q1_text:
        variants.append(QueryVariant(
            query_id="Q1",
            text=q1_text,
            strategy="anchor_tight",
        ))

    # Q2: anchor-medium (entities + seed terms, no date)
    q2_parts = entities + seed_terms[:4]
    q2_text = _truncate_query(" ".join(q2_parts), max_len)
    # Only add if different from Q1
    if q2_text and (not variants or q2_text != variants[0].text):
        variants.append(QueryVariant(
            query_id="Q2",
            text=q2_text,
            strategy="anchor_medium",
        ))

    # Q3: broad (entities + 2-3 seed terms)
    q3_parts = entities[:2] + seed_terms[:3]
    q3_text = _truncate_query(" ".join(q3_parts), max_len)
    # Only add if different from existing variants
    existing_texts = {v.text for v in variants}
    if q3_text and q3_text not in existing_texts:
        variants.append(QueryVariant(
            query_id="Q3",
            text=q3_text,
            strategy="broad",
        ))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# Topic Selection
# ─────────────────────────────────────────────────────────────────────────────


# Falsifiable_by values that indicate news topic
NEWS_FALSIFIABLE_BY = {"reputable_news", "official_statement"}

# Falsifiable_by values that depend on time_anchor
TIME_DEPENDENT_FALSIFIABLE_BY = {"dataset", "scientific_publication", "public_records"}


def select_topic_from_claim(claim: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Select Tavily topic from structured claim fields.

    Returns: (topic, reason_codes)

    Rules:
    - falsifiable_by in {reputable_news, official_statement} → "news"
    - falsifiable_by in {dataset, scientific_publication, public_records}:
      - time_anchor.type in {explicit_date, range} → "news"
      - otherwise → "general"
    - default → "news"
    """
    reason_codes: list[str] = []

    # Extract falsifiability info
    falsifiability = claim.get("falsifiability")
    if isinstance(falsifiability, dict):
        falsifiable_by = falsifiability.get("falsifiable_by", "other")
    else:
        falsifiable_by = "other"
        reason_codes.append("no_falsifiability")

    # Rule 1: News sources
    if falsifiable_by in NEWS_FALSIFIABLE_BY:
        reason_codes.append(f"falsifiable_by:{falsifiable_by}")
        return "news", reason_codes

    # Rule 2: Time-dependent sources
    if falsifiable_by in TIME_DEPENDENT_FALSIFIABLE_BY:
        time_anchor = claim.get("time_anchor")
        time_type = "unknown"
        if isinstance(time_anchor, dict):
            time_type = time_anchor.get("type", "unknown")

        reason_codes.append(f"falsifiable_by:{falsifiable_by}")
        reason_codes.append(f"time_anchor:{time_type}")

        if time_type in ("explicit_date", "range"):
            return "news", reason_codes
        else:
            return "general", reason_codes

    # Default: news (better for recent events and domain allowlists)
    reason_codes.append("default_news")
    return "news", reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Outcome
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalOutcome:
    """Observable quality signals from search results."""

    sources_count: int
    """Total number of sources returned."""

    best_relevance: float
    """Highest relevance score among sources."""

    usable_snippets_count: int
    """Number of sources with non-empty snippet/content."""

    match_accept_count: int
    """Sources with quote_matches=True or stance in (SUPPORT, REFUTE)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_count": self.sources_count,
            "best_relevance": self.best_relevance,
            "usable_snippets_count": self.usable_snippets_count,
            "match_accept_count": self.match_accept_count,
        }


ACCEPTABLE_STANCES = {"support", "refute", "SUPPORT", "REFUTE"}


def compute_retrieval_outcome(sources: list[dict[str, Any]]) -> RetrievalOutcome:
    """Compute observable quality signals from search results."""
    sources_count = len(sources)
    best_relevance = 0.0
    usable_snippets_count = 0
    match_accept_count = 0

    for src in sources:
        if not isinstance(src, dict):
            continue

        # Track best relevance
        score = src.get("score") or src.get("relevance_score") or 0.0
        if isinstance(score, (int, float)) and score > best_relevance:
            best_relevance = float(score)

        # Count usable snippets
        snippet = src.get("snippet") or src.get("content") or src.get("raw_content")
        if isinstance(snippet, str) and len(snippet) >= 50:
            usable_snippets_count += 1

        # Count match accepts
        quote_matches = src.get("quote_matches", False)
        stance = src.get("stance", "")
        if quote_matches or stance in ACCEPTABLE_STANCES:
            match_accept_count += 1

    return RetrievalOutcome(
        sources_count=sources_count,
        best_relevance=best_relevance,
        usable_snippets_count=usable_snippets_count,
        match_accept_count=match_accept_count,
    )


def should_stop_escalation(
    outcome: RetrievalOutcome,
    config: EscalationConfig | None = None,
) -> tuple[bool, str]:
    """
    Determine if escalation should stop based on outcome.

    Returns: (should_stop, reason)

    Stop if:
    - match_accept_count >= 1
    - usable_snippets_count >= min_usable_snippets AND best_relevance >= min_relevance_threshold
    """
    cfg = config or DEFAULT_ESCALATION_CONFIG

    if outcome.match_accept_count >= 1:
        return True, "match_accept"

    if (
        outcome.usable_snippets_count >= cfg.min_usable_snippets
        and outcome.best_relevance >= cfg.min_relevance_threshold
    ):
        return True, "snippets_and_relevance"

    return False, "continue"


# ─────────────────────────────────────────────────────────────────────────────
# Escalation Ladder
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EscalationPass:
    """Configuration for one escalation pass."""

    pass_id: str
    """Pass identifier: 'A', 'B', 'C', 'D'."""

    search_depth: str
    """Tavily search depth: 'basic' or 'advanced'."""

    max_results: int
    """Maximum results to request."""

    topic: str | None
    """Override topic, or None to use claim-derived topic."""

    include_domains_relaxed: bool
    """If True, remove include_domains restriction."""

    query_ids: list[str]
    """Which query variants to try: ['Q1', 'Q2'] etc."""

    trigger_conditions: list[str] = field(default_factory=list)
    """Conditions that trigger this pass (for documentation)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "topic": self.topic,
            "include_domains_relaxed": self.include_domains_relaxed,
            "query_ids": self.query_ids,
        }


def get_escalation_ladder() -> list[EscalationPass]:
    """
    Return the 4-pass escalation ladder.

    Pass A (cheap): basic depth, 3 results, Q1 then Q2
    Pass B (expand): basic depth, 6 results, Q2 then Q3
    Pass C (deep): advanced depth, 6 results, Q1 then Q2
    Pass D (domain relaxation): basic depth, 6 results, Q2 only, domains relaxed
    """
    return [
        EscalationPass(
            pass_id="A",
            search_depth="basic",
            max_results=3,
            topic=None,  # Use claim-derived topic
            include_domains_relaxed=False,
            query_ids=["Q1", "Q2"],
            trigger_conditions=["initial"],
        ),
        EscalationPass(
            pass_id="B",
            search_depth="basic",
            max_results=6,
            topic=None,
            include_domains_relaxed=False,
            query_ids=["Q2", "Q3"],
            trigger_conditions=["no_snippets", "no_matches"],
        ),
        EscalationPass(
            pass_id="C",
            search_depth="advanced",
            max_results=6,
            topic=None,
            include_domains_relaxed=False,
            query_ids=["Q1", "Q2"],
            trigger_conditions=["no_evidence_after_B"],
        ),
        EscalationPass(
            pass_id="D",
            search_depth="basic",
            max_results=6,
            topic=None,
            include_domains_relaxed=True,
            query_ids=["Q2"],
            trigger_conditions=["domain_mismatch", "last_resort"],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Escalation Reason Codes
# ─────────────────────────────────────────────────────────────────────────────


def compute_escalation_reason_codes(outcome: RetrievalOutcome) -> list[str]:
    """Compute reason codes for why escalation is needed."""
    reasons: list[str] = []

    if outcome.sources_count == 0:
        reasons.append("no_sources")
    if outcome.usable_snippets_count == 0:
        reasons.append("no_snippets")
    if outcome.match_accept_count == 0:
        reasons.append("no_matches")
    if outcome.best_relevance < 0.2:
        reasons.append("low_relevance")

    return reasons if reasons else ["insufficient_quality"]


# ─────────────────────────────────────────────────────────────────────────────
# Trace Events
# ─────────────────────────────────────────────────────────────────────────────


def trace_query_variants(claim_id: str, variants: list[QueryVariant]) -> None:
    """Log query variants trace event."""
    Trace.event(
        "search.query.variants",
        {
            "claim_id": claim_id,
            "variants": [v.to_dict() for v in variants],
            "count": len(variants),
        },
    )


def trace_topic_selected(claim_id: str, topic: str, reason_codes: list[str]) -> None:
    """Log topic selection trace event."""
    Trace.event(
        "search.topic.selected",
        {
            "claim_id": claim_id,
            "topic": topic,
            "reason_codes": reason_codes,
        },
    )


def trace_escalation_pass(
    claim_id: str,
    pass_config: EscalationPass,
    query_id: str,
    reason_codes: list[str],
    outcome: RetrievalOutcome,
    include_domains_count: int | None = None,
) -> None:
    """Log escalation pass trace event."""
    Trace.event(
        "search.escalation",
        {
            "claim_id": claim_id,
            "pass_id": pass_config.pass_id,
            "query_id": query_id,
            "reason_codes": reason_codes,
            "params": {
                "search_depth": pass_config.search_depth,
                "max_results": pass_config.max_results,
                "include_domains_relaxed": pass_config.include_domains_relaxed,
                "include_domains_count": include_domains_count,
            },
            "outcome": outcome.to_dict(),
        },
    )


def trace_search_stop(
    claim_id: str,
    pass_id: str,
    stop_reason: str,
    outcome: RetrievalOutcome,
) -> None:
    """Log early stop trace event."""
    Trace.event(
        "search.stop",
        {
            "claim_id": claim_id,
            "pass_id": pass_id,
            "stop_reason": stop_reason,
            "outcome": outcome.to_dict(),
        },
    )


def trace_search_summary(
    claim_id: str,
    passes_executed: int,
    tavily_calls: int,
    final_outcome: RetrievalOutcome,
    domains_relaxed: bool,
) -> None:
    """Log end-of-retrieval summary trace event."""
    Trace.event(
        "search.summary",
        {
            "claim_id": claim_id,
            "passes_executed": passes_executed,
            "tavily_calls": tavily_calls,
            "best_relevance_final": final_outcome.best_relevance,
            "usable_snippets_final": final_outcome.usable_snippets_count,
            "match_accept_final": final_outcome.match_accept_count,
            "domains_relaxed": domains_relaxed,
        },
    )
