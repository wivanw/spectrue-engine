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
Search Escalation Policy

Evidence-driven, deterministic escalation for Tavily retrieval.
Start cheap, expand only when observable quality signals indicate need.

Key components:
- QueryVariant: Deterministic query variants from claim fields (Q1/Q2/Q3)
- EscalationPass: Multi-pass search with escalating parameters (A→B→C→D)
- RetrievalOutcome: Observable RETRIEVAL-ONLY signals (no post-match fields)
- Topic selection from structured claim fields (no text heuristics)

IMPORTANT: RetrievalOutcome uses ONLY signals available from raw Tavily results:
- sources_count: number of results
- best_relevance: Tavily's relevance/score field
- usable_snippets_count: results with non-trivial snippet text

Do NOT include post-match signals like quote_matches/stance - those are set by
matcher/judge and not available at retrieval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.verification.types import SearchDepth
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.orchestration.sufficiency import check_sufficiency_for_claim, SufficiencyStatus


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EscalationConfig:
    """Configuration for escalation thresholds.
    
    All magic numbers live here for traceability.
    """

    min_relevance_threshold: float = 0.2
    """Minimum relevance score to consider evidence usable."""

    min_usable_snippets: int = 2
    """Minimum snippets needed for early stop (combined with relevance)."""

    max_query_length: int = 80
    """Maximum length for query variants."""

    min_snippet_chars: int = 50
    """Minimum characters for a snippet to be considered usable."""

    max_token_length: int = 30
    """Maximum characters for a single token (filter out garbage)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_relevance_threshold": self.min_relevance_threshold,
            "min_usable_snippets": self.min_usable_snippets,
            "max_query_length": self.max_query_length,
            "min_snippet_chars": self.min_snippet_chars,
            "max_token_length": self.max_token_length,
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


def _normalize_token(token: str) -> str:
    """Normalize a token: lowercase, strip whitespace."""
    return token.strip().lower()


def _is_valid_token(token: str, max_length: int = 30) -> bool:
    """Check if token is valid (not empty, not too long, not too short)."""
    if not token or len(token) < 2:
        return False
    if len(token) > max_length:
        return False
    # Reject tokens that are mostly numbers (likely IDs/codes)
    if sum(c.isdigit() for c in token) > len(token) * 0.7:
        return False
    return True


def _deduplicate_tokens(tokens: list[str], max_token_length: int = 30) -> list[str]:
    """Deduplicate tokens preserving order, with normalization and filtering."""
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        normalized = _normalize_token(token)
        if not _is_valid_token(normalized, max_token_length):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(token)  # Keep original case
    return result


def _extract_top_entities(
    claim: dict[str, Any], max_count: int = 3, max_token_length: int = 30
) -> list[str]:
    """Extract top subject entities from claim."""
    entities = claim.get("subject_entities", [])
    if not isinstance(entities, list):
        return []
    valid: list[str] = []
    for e in entities:
        if not isinstance(e, str):
            continue
        # Reject entities that are too long or too short
        if len(e) < 2 or len(e) > max_token_length:
            continue
        valid.append(e)
        if len(valid) >= max_count:
            break
    return valid


def _extract_seed_terms(
    claim: dict[str, Any], max_count: int = 6, max_token_length: int = 30
) -> list[str]:
    """Extract retrieval seed terms from claim.
    
    Filters out multi-word phrases (>3 words) to keep queries keyword-like.
    """
    terms = claim.get("retrieval_seed_terms", [])
    if not isinstance(terms, list):
        return []
    valid: list[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        t_stripped = t.strip()
        # Reject terms that are too short
        if len(t_stripped) < 2:
            continue
        # Reject multi-word phrases (>3 words) - these make queries sentence-like
        word_count = len(t_stripped.split())
        if word_count > 3:
            continue
        # Reject very long single tokens
        if word_count == 1 and len(t_stripped) > max_token_length:
            continue
        valid.append(t_stripped)
        if len(valid) >= max_count:
            break
    return valid


def _extract_context_entities(
    claim: dict[str, Any], max_count: int = 2, max_token_length: int = 30
) -> list[str]:
    """Extract context entities from claim.
    
    Context entities are inherited from document-level pool during skeleton→claims conversion.
    They provide additional query terms for contextless claims.
    """
    entities = claim.get("context_entities", [])
    if not isinstance(entities, list):
        return []
    valid: list[str] = []
    for e in entities:
        if not isinstance(e, str):
            continue
        e_stripped = e.strip()
        # Reject entities that are too long or too short
        if len(e_stripped) < 2 or len(e_stripped) > max_token_length:
            continue
        valid.append(e_stripped)
        if len(valid) >= max_count:
            break
    return valid


def _extract_date_anchor(claim: dict[str, Any]) -> str | None:
    """Extract explicit date from time_anchor if available.
    
    For 'range' type, returns only the start date to avoid query pollution.
    """
    time_anchor = claim.get("time_anchor")
    if not isinstance(time_anchor, dict):
        return None

    anchor_type = time_anchor.get("type", "unknown")
    if anchor_type == "explicit_date":
        date_val = time_anchor.get("value")
        if isinstance(date_val, str) and len(date_val) >= 4:
            # Return date portion only (YYYY-MM-DD or YYYY)
            return date_val[:10]
    elif anchor_type == "range":
        # For range, use start date only (end date would bloat query)
        start_val = time_anchor.get("start")
        if isinstance(start_val, str) and len(start_val) >= 4:
            return start_val[:10]
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

    Q1 (anchor-tight): <top entities> <context entities> <top seed terms> <date/range if explicit>
    Q2 (anchor-medium): <top entities> <context entities> <top seed terms>
    Q3 (broad): <top entities> <2-3 seed terms>

    Context entities (from document pool) are included in Q1/Q2 for contextless claims.
    Each query <= max_query_length chars, keyword-like (no sentences).
    Tokens are deduplicated and normalized.
    """
    cfg = config or DEFAULT_ESCALATION_CONFIG
    max_len = cfg.max_query_length
    max_token_len = cfg.max_token_length

    entities = _extract_top_entities(claim, max_count=3, max_token_length=max_token_len)
    context_entities = _extract_context_entities(claim, max_count=2, max_token_length=max_token_len)
    seed_terms = _extract_seed_terms(claim, max_count=6, max_token_length=max_token_len)
    date_anchor = _extract_date_anchor(claim)

    # Early return if no query can be built (empty query guard)
    if not entities and not seed_terms and not context_entities:
        Trace.event("search.query.empty_blocked", {
            "claim_id": claim.get("id", "unknown"),
            "reason": "no_queryable_terms",
        })
        return []

    variants: list[QueryVariant] = []

    # Q1: anchor-tight (entities + context + seed terms + date)
    q1_parts = _deduplicate_tokens(entities + context_entities + seed_terms[:4], max_token_len)
    if date_anchor:
        q1_parts.append(date_anchor)
    q1_text = _truncate_query(" ".join(q1_parts), max_len)
    if q1_text:
        variants.append(QueryVariant(
            query_id="Q1",
            text=q1_text,
            strategy="anchor_tight",
        ))

    # Q2: anchor-medium (entities + context + seed terms, no date)
    q2_parts = _deduplicate_tokens(entities + context_entities + seed_terms[:4], max_token_len)
    q2_text = _truncate_query(" ".join(q2_parts), max_len)
    # Only add if different from Q1
    if q2_text and (not variants or q2_text != variants[0].text):
        variants.append(QueryVariant(
            query_id="Q2",
            text=q2_text,
            strategy="anchor_medium",
        ))

    # Q3: broad (fewer terms, no context - keeps it very short)
    # If entities empty, use first few seed terms only
    if entities:
        q3_parts = _deduplicate_tokens(entities[:2] + seed_terms[:2], max_token_len)
    else:
        # No entities → use more seed terms but still short
        q3_parts = _deduplicate_tokens(seed_terms[:3], max_token_len)
    
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
    if not isinstance(falsifiability, dict):
        # No falsifiability field at all
        reason_codes.append("no_falsifiability_field")
        reason_codes.append("default_news")
        return "news", reason_codes
    
    falsifiable_by = falsifiability.get("falsifiable_by", "other")
    
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
    # Provide diagnostic info about why we're defaulting
    reason_codes.append(f"falsifiable_by:{falsifiable_by}")
    reason_codes.append("unclassified_falsifiable_by")
    reason_codes.append("default_news")
    return "news", reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Outcome
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalOutcome:
    """Observable quality signals from RAW search results.
    
    IMPORTANT: This is for RETRIEVAL signals only - fields that exist in
    raw Tavily results. Do NOT include post-match signals like quote_matches
    or stance, which are added by matcher/judge after retrieval.
    """

    sources_count: int
    """Total number of sources returned."""

    best_relevance: float
    """Highest relevance score among sources (Tavily's score field)."""

    usable_snippets_count: int
    """Number of sources with snippet >= min_snippet_chars."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_count": self.sources_count,
            "best_relevance": self.best_relevance,
            "usable_snippets_count": self.usable_snippets_count,
        }


def compute_retrieval_outcome(
    sources: list[dict[str, Any]],
    config: EscalationConfig | None = None,
) -> RetrievalOutcome:
    """Compute observable quality signals from RAW search results.
    
    Uses only Tavily-provided fields: score/relevance_score, snippet/content.
    """
    cfg = config or DEFAULT_ESCALATION_CONFIG
    min_snippet_len = cfg.min_snippet_chars
    
    sources_count = len(sources)
    best_relevance = 0.0
    usable_snippets_count = 0

    for src in sources:
        if not isinstance(src, dict):
            continue

        # Track best relevance (Tavily's score field)
        score = src.get("score") or src.get("relevance_score") or 0.0
        if isinstance(score, (int, float)) and score > best_relevance:
            best_relevance = float(score)

        # Count usable snippets (configurable threshold)
        snippet = src.get("snippet") or src.get("content") or src.get("raw_content")
        if isinstance(snippet, str) and len(snippet) >= min_snippet_len:
            usable_snippets_count += 1

    return RetrievalOutcome(
        sources_count=sources_count,
        best_relevance=best_relevance,
        usable_snippets_count=usable_snippets_count,
    )


def should_stop_escalation(
    claim: dict[str, Any],
    sources: list[dict[str, Any]],
    config: EscalationConfig | None = None,
) -> tuple[bool, str]:
    """
    Determine if escalation should stop based on Bayesian sufficiency.

    Returns: (should_stop, reason)
    """
    # Use the unified Bayesian judge
    sufficiency = check_sufficiency_for_claim(claim, sources)
    
    if sufficiency.status == SufficiencyStatus.SUFFICIENT:
        return True, f"bayesian_sufficiency: {sufficiency.rule_matched}"

    return False, "insufficient"


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
    """If True, remove include_domains restriction (set to None)."""

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
    Pass D (domain relaxation): basic depth, 6 results, Q2 only, include_domains=None
    """
    return [
        EscalationPass(
            pass_id="A",
            search_depth=SearchDepth.BASIC.value,
            max_results=3,
            topic=None,  # Use claim-derived topic
            include_domains_relaxed=False,
            query_ids=["Q1", "Q2"],
            trigger_conditions=["initial"],
        ),
        EscalationPass(
            pass_id="B",
            search_depth=SearchDepth.BASIC.value,
            max_results=5,
            topic=None,
            include_domains_relaxed=False,
            query_ids=["Q2", "Q3"],
            trigger_conditions=["no_snippets", "low_relevance"],
        ),
        EscalationPass(
            pass_id="C",
            search_depth=SearchDepth.ADVANCED.value,
            max_results=5,
            topic=None,
            include_domains_relaxed=False,
            query_ids=["Q1", "Q2"],
            trigger_conditions=["no_evidence_after_B"],
        ),
        EscalationPass(
            pass_id="D",
            search_depth=SearchDepth.BASIC.value,
            max_results=5,
            topic=None,
            include_domains_relaxed=True,  # include_domains will be None
            query_ids=["Q2"],
            trigger_conditions=["domain_mismatch", "last_resort"],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Escalation Reason Codes
# ─────────────────────────────────────────────────────────────────────────────


def compute_escalation_reason_codes(
    claim: dict[str, Any],
    sources: list[dict[str, Any]],
    outcome: RetrievalOutcome,
    config: EscalationConfig | None = None,
) -> list[str]:
    """Compute reason codes for why escalation is needed."""
    sufficiency = check_sufficiency_for_claim(claim, sources)
    
    reasons: list[str] = []
    if outcome.sources_count == 0:
        reasons.append("no_sources")
    
    # Add the descriptive reason from the Bayesian judge
    reasons.append(f"bayesian:{sufficiency.reason}")
    
    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Trace Events
# ─────────────────────────────────────────────────────────────────────────────


def trace_query_variants(
    claim_id: str, 
    variants: list[QueryVariant],
    claim: dict[str, Any] | None = None,
) -> None:
    """Log query variants trace event.
    
    Args:
        claim_id: The claim ID
        variants: Built query variants
        claim: Optional claim dict to extract context entities count
    """
    context_count = 0
    if claim:
        context_entities = claim.get("context_entities", [])
        if isinstance(context_entities, list):
            context_count = len(context_entities)
    
    Trace.event(
        "search.query.variants",
        {
            "claim_id": claim_id,
            "variants": [v.to_dict() for v in variants],
            "count": len(variants),
            "included_context_entities_count": context_count,
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
    """Log escalation pass trace event.
    
    include_domains_count should be:
    - int: number of domains in allowlist
    - None: when include_domains_relaxed=True (no allowlist)
    """
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
            "domains_relaxed": domains_relaxed,
        },
    )
