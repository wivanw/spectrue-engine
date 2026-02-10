# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.domain.verification.search.types import SearchDepth
from spectrue_core.domain.verification.search.escalation_model import (
    EscalationConfig,
    QueryVariant,
    RetrievalOutcome,
    EscalationPass,
)
from spectrue_core.domain.claims.sufficiency import SufficiencyStatus
from spectrue_core.use_cases.claims.sufficiency import check_sufficiency_for_claim
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

DEFAULT_ESCALATION_CONFIG = EscalationConfig()

# Falsifiable_by values that indicate news topic
NEWS_FALSIFIABLE_BY = {"reputable_news", "official_statement"}

# Falsifiable_by values that depend on time_anchor
TIME_DEPENDENT_FALSIFIABLE_BY = {"dataset", "scientific_publication", "public_records"}


def _normalize_token(token: str) -> str:
    """Normalize a token: lowercase, strip whitespace."""
    return token.strip().lower()


def _is_valid_token(token: str, max_length: int = 30) -> bool:
    """Check if token is valid (not empty, not too long, not too short)."""
    if not token or len(token) < 2:
        return False
    if len(token) > max_length:
        return False
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
        result.append(token)
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
        if len(e) < 2 or len(e) > max_token_length:
            continue
        valid.append(e)
        if len(valid) >= max_count:
            break
    return valid


def _extract_seed_terms(
    claim: dict[str, Any], max_count: int = 6, max_token_length: int = 30
) -> list[str]:
    """Extract retrieval seed terms from claim."""
    terms = claim.get("retrieval_seed_terms", [])
    if not isinstance(terms, list):
        return []
    valid: list[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        t_stripped = t.strip()
        if len(t_stripped) < 2:
            continue
        word_count = len(t_stripped.split())
        if word_count > 3:
            continue
        if word_count == 1 and len(t_stripped) > max_token_length:
            continue
        valid.append(t_stripped)
        if len(valid) >= max_count:
            break
    return valid


def _extract_context_entities(
    claim: dict[str, Any], max_count: int = 2, max_token_length: int = 30
) -> list[str]:
    """Extract context entities from claim."""
    entities = claim.get("context_entities", [])
    if not isinstance(entities, list):
        return []
    valid: list[str] = []
    for e in entities:
        if not isinstance(e, str):
            continue
        e_stripped = e.strip()
        if len(e_stripped) < 2 or len(e_stripped) > max_token_length:
            continue
        valid.append(e_stripped)
        if len(valid) >= max_count:
            break
    return valid


def _extract_date_anchor(claim: dict[str, Any]) -> str | None:
    """Extract explicit date from time_anchor if available."""
    time_anchor = claim.get("time_anchor")
    if not isinstance(time_anchor, dict):
        return None

    anchor_type = time_anchor.get("type", "unknown")
    if anchor_type == "explicit_date":
        date_val = time_anchor.get("value")
        if isinstance(date_val, str) and len(date_val) >= 4:
            return date_val[:10]
    elif anchor_type == "range":
        start_val = time_anchor.get("start")
        if isinstance(start_val, str) and len(start_val) >= 4:
            return start_val[:10]
    return None


def _truncate_query(query: str, max_length: int) -> str:
    """Truncate query to max_length, respecting word boundaries."""
    if len(query) <= max_length:
        return query
    truncated = query[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space].strip()
    return truncated.strip()


def build_query_variants(
    claim: dict[str, Any],
    config: EscalationConfig | None = None,
) -> list[QueryVariant]:
    """Build up to 3 query variants from claim structured fields."""
    cfg = config or DEFAULT_ESCALATION_CONFIG
    max_len = cfg.max_query_length
    max_token_len = cfg.max_token_length

    entities = _extract_top_entities(claim, max_count=3, max_token_length=max_token_len)
    context_entities = _extract_context_entities(claim, max_count=2, max_token_length=max_token_len)
    seed_terms = _extract_seed_terms(claim, max_count=6, max_token_length=max_token_len)
    date_anchor = _extract_date_anchor(claim)

    if not entities and not seed_terms and not context_entities:
        Trace.event("search.query.empty_blocked", {
            "claim_id": claim.get("id", "unknown"),
            "reason": "no_queryable_terms",
        })
        return []

    variants: list[QueryVariant] = []

    # Q1: anchor-tight
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

    # Q2: anchor-medium
    q2_parts = _deduplicate_tokens(entities + context_entities + seed_terms[:4], max_token_len)
    q2_text = _truncate_query(" ".join(q2_parts), max_len)
    if q2_text and (not variants or q2_text != variants[0].text):
        variants.append(QueryVariant(
            query_id="Q2",
            text=q2_text,
            strategy="anchor_medium",
        ))

    # Q3: broad
    if entities:
        q3_parts = _deduplicate_tokens(entities[:2] + seed_terms[:2], max_token_len)
    else:
        q3_parts = _deduplicate_tokens(seed_terms[:3], max_token_len)
    
    q3_text = _truncate_query(" ".join(q3_parts), max_len)
    existing_texts = {v.text for v in variants}
    if q3_text and q3_text not in existing_texts:
        variants.append(QueryVariant(
            query_id="Q3",
            text=q3_text,
            strategy="broad",
        ))

    return variants


def select_topic_from_claim(claim: dict[str, Any]) -> tuple[str, list[str]]:
    """Select Tavily topic from structured claim fields."""
    reason_codes: list[str] = []

    falsifiability = claim.get("falsifiability")
    if not isinstance(falsifiability, dict):
        reason_codes.append("no_falsifiability_field")
        reason_codes.append("default_news")
        return "news", reason_codes
    
    falsifiable_by = falsifiability.get("falsifiable_by", "other")
    
    if falsifiable_by in NEWS_FALSIFIABLE_BY:
        reason_codes.append(f"falsifiable_by:{falsifiable_by}")
        return "news", reason_codes

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

    reason_codes.append(f"falsifiable_by:{falsifiable_by}")
    reason_codes.append("unclassified_falsifiable_by")
    reason_codes.append("default_news")
    return "news", reason_codes


def compute_retrieval_outcome(
    sources: list[dict[str, Any]],
    config: EscalationConfig | None = None,
) -> RetrievalOutcome:
    """Compute observable quality signals from RAW search results."""
    cfg = config or DEFAULT_ESCALATION_CONFIG
    min_snippet_len = cfg.min_snippet_chars
    
    sources_count = len(sources)
    best_relevance = 0.0
    usable_snippets_count = 0

    for src in sources:
        if not isinstance(src, dict):
            continue

        score = src.get("score") or src.get("relevance_score") or 0.0
        if isinstance(score, (int, float)) and score > best_relevance:
            best_relevance = float(score)

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
    """Determine if escalation should stop based on Bayesian sufficiency."""
    cfg = config or DEFAULT_ESCALATION_CONFIG
    
    sufficiency = check_sufficiency_for_claim(claim, sources)
    if sufficiency.status == SufficiencyStatus.SUFFICIENT:
        return True, f"bayesian_sufficiency:{sufficiency.rule_matched}"

    outcome = compute_retrieval_outcome(sources, cfg)
    if outcome.usable_snippets_count >= cfg.min_usable_snippets and outcome.best_relevance >= cfg.min_relevance_threshold:
        return True, "snippets_and_relevance"

    return False, "insufficient"


def get_escalation_ladder() -> list[EscalationPass]:
    """Return the 4-pass escalation ladder."""
    return [
        EscalationPass(
            pass_id="A",
            search_depth=SearchDepth.BASIC.value,
            max_results=3,
            topic=None,
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
            include_domains_relaxed=True,
            query_ids=["Q2"],
            trigger_conditions=["domain_mismatch", "last_resort"],
        ),
    ]


def compute_escalation_reason_codes(
    claim: dict[str, Any],
    sources: list[dict[str, Any]],
    outcome: RetrievalOutcome,
    config: EscalationConfig | None = None,
) -> list[str]:
    """Compute reason codes for why escalation is needed."""
    cfg = config or DEFAULT_ESCALATION_CONFIG
    sufficiency = check_sufficiency_for_claim(claim, sources)
    
    reasons: list[str] = []
    if outcome.sources_count == 0:
        reasons.append("no_sources")
    
    if outcome.sources_count > 0 and outcome.usable_snippets_count == 0:
        reasons.append("no_snippets")
    if outcome.best_relevance < cfg.min_relevance_threshold:
        reasons.append(f"low_relevance(threshold={cfg.min_relevance_threshold})")
    if outcome.usable_snippets_count < cfg.min_usable_snippets:
        reasons.append(f"insufficient_snippets({outcome.usable_snippets_count}<{cfg.min_usable_snippets})")

    reasons.append(f"bayesian:{sufficiency.reason}")
    return reasons


def trace_query_variants(
    claim_id: str, 
    variants: list[QueryVariant],
    claim: dict[str, Any] | None = None,
) -> None:
    """Log query variants trace event."""
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
            "domains_relaxed": domains_relaxed,
        },
    )
