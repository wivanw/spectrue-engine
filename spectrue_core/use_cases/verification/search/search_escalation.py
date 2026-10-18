# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.domain.verification.search.types import SearchDepth
from spectrue_core.domain.verification.search.escalation_model import (
    EscalationConfig,
    RetrievalOutcome,
    EscalationPass,
)
from spectrue_core.domain.claims.sufficiency import SufficiencyStatus
from spectrue_core.use_cases.claims.sufficiency import check_sufficiency_for_claim
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

DEFAULT_ESCALATION_CONFIG = EscalationConfig()


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
