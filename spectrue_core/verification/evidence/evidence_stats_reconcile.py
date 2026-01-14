from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from spectrue_core.scoring.budget_allocation import GlobalBudgetTracker
from spectrue_core.utils.trace import Trace


def reconcile_budget_state_from_sources(
    tracker: GlobalBudgetTracker,
    sources: Iterable[dict[str, Any]],
    *,
    context: str,
    claim_id: str | None,
) -> None:
    """
    Reconcile BudgetState counters from the actual post-enrichment sources.

    Motivation:
    - In cache_only / snippet-only paths, record_extract() is never called.
      That leaves BudgetState.total_sources/relevant_sources at 0 even when
      quote enrichment succeeded (quotes_added > 0).
    - Downstream explainability (A) relies on these counters (directly or via
      evidence_sufficiency/diversity), so inconsistent counters depress A.

    This function is intentionally *purely aggregational*:
    - It does NOT modify extracts_used (cost) and does NOT touch alpha/beta.
    - It only reconciles observable counters and derived relevance aggregates.
    """
    src_list = list(sources or [])
    st = tracker.state

    # Reset observable counters (do not touch extracts_used / alpha / beta)
    st.total_sources = 0
    st.relevant_sources = 0
    st.quotes_found = 0
    st.sum_relevance = 0.0
    st.max_relevance = 0.0
    st.authoritative_count = 0

    for src in src_list:
        # Count every candidate as a "source observed" (this is what A needs),
        # regardless of whether it was fetched or came from snippet/cache.
        st.total_sources += 1

        relevance = src.get("relevance_score", 0.5)
        try:
            relevance_f = float(relevance)
        except Exception:
            relevance_f = 0.5

        st.sum_relevance += relevance_f
        st.max_relevance = max(st.max_relevance, relevance_f)

        if relevance_f >= 0.6:
            st.relevant_sources += 1

        if src.get("quote"):
            st.quotes_found += 1

        tier = (src.get("source_tier") or "").upper()
        if tier in ("A", "B"):
            st.authoritative_count += 1

    Trace.event(
        "evidence.stats.reconciled",
        {
            "context": context,
            "claim_id": claim_id,
            "sources": len(src_list),
            "total_sources": st.total_sources,
            "relevant_sources": st.relevant_sources,
            "quotes_found": st.quotes_found,
            "max_relevance": round(st.max_relevance, 4),
            "avg_relevance": round(st.avg_relevance, 4),
        },
    )
