"""
Budget reconciliation logic.

Ensures that sources found via snippets/cache are counted in the budget state.
"""
from typing import Any

from spectrue_core.scoring.budget_allocation import GlobalBudgetTracker
from spectrue_core.utils.source_utils import has_evidence_chunk

def reconcile_budget_state_from_sources(
    tracker: GlobalBudgetTracker,
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    claim_id: str,
) -> None:
    """
    Reconcile evidence counters for budget tracking.
    
    Ensures that sources found via snippets/cache are counted in the budget state,
    even if we didn't explicitly fetch/transcribe them in the current ladder step.
    """
    for src in candidates:
        if src.get("_budget_observed"):
            continue
            
        # Check if source is useful (has quote/snippet or high relevance)
        relevance = float(src.get("relevance", 0.0))
        has_chunk = has_evidence_chunk(src)
        is_authoritative = bool(src.get("is_authoritative"))
        
        # Only count if it contributes to belief
        # Threshold 0.4 matches heuristic for "potentially useful"
        if relevance > 0.4 or has_chunk or is_authoritative:
            tracker.record_extract(
                relevance_score=relevance,
                has_quote=has_chunk,
                is_authoritative=is_authoritative,
            )
            src["_budget_observed"] = True
