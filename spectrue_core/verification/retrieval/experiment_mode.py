# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Retrieval Experiment Mode helpers.

This module provides functions and context for controlling experiment mode behavior.
When experiment mode is enabled:
- Sanity gate bypass: ALL sources kept regardless of overlap/score
- Clustering bypass: ALL kept sources passed to extraction (no reps-only)
- Budget bypass: Budgets treated as infinite (except API max_results)
- Source broadcasting: ALL sources passed to ALL claims
- No early-stop: Escalation runs full ladder depth

Toggle via: SPECTRUE_RETRIEVAL_EXPERIMENT_MODE=1
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from spectrue_core.utils.trace import Trace

# Context variable for experiment mode (allows per-request override)
_experiment_mode_ctx: ContextVar[bool | None] = ContextVar(
    "retrieval_experiment_mode", default=None
)

# Track if we've already emitted the startup trace
_startup_traced: bool = False


def is_experiment_mode() -> bool:
    """
    Check if retrieval experiment mode is enabled.
    
    Priority:
    1. Context variable (per-request override)
    2. Environment variable SPECTRUE_RETRIEVAL_EXPERIMENT_MODE
    
    Returns:
        True if experiment mode is enabled
    """
    # Check context variable first (allows per-request override)
    ctx_value = _experiment_mode_ctx.get()
    if ctx_value is not None:
        return ctx_value
    
    # Fall back to env var
    raw = os.getenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def set_experiment_mode(enabled: bool) -> None:
    """
    Set experiment mode for the current async context.
    
    Use this to override experiment mode for specific requests without
    changing the environment variable.
    
    Args:
        enabled: Whether to enable experiment mode
    """
    _experiment_mode_ctx.set(enabled)


def reset_experiment_mode() -> None:
    """Reset experiment mode context variable to default (use env var)."""
    _experiment_mode_ctx.set(None)


def should_filter_sources() -> bool:
    """
    Check if sources should be filtered (sanity gate, clustering).
    
    Returns:
        True if filtering should be applied (normal mode)
        False if filtering should be bypassed (experiment mode)
    """
    return not is_experiment_mode()


def should_apply_budget() -> bool:
    """
    Check if budget limits should be applied.
    
    Returns:
        True if budget limits should be enforced (normal mode)
        False if budgets should be treated as infinite (experiment mode)
    """
    return not is_experiment_mode()


def should_stop_early() -> bool:
    """
    Check if escalation should stop early on "good enough" evidence.
    
    Returns:
        True if early stopping is allowed (normal mode)
        False if full escalation ladder should run (experiment mode)
    """
    return not is_experiment_mode()


def emit_experiment_mode_trace() -> None:
    """
    Emit trace event for experiment mode status (once per process).
    
    Call this during retrieval initialization to log experiment mode state.
    """
    global _startup_traced
    if _startup_traced:
        return
    
    enabled = is_experiment_mode()
    Trace.event("retrieval.experiment_mode", {"enabled": enabled})
    _startup_traced = True


def emit_filters_bypassed_trace() -> None:
    """
    Emit trace event when filters are bypassed in experiment mode.
    """
    if not is_experiment_mode():
        return
    
    Trace.event("retrieval.filters.bypassed", {
        "sanity_gate": True,
        "clustering_reps_only": True,
        "mismatch_block": True,
    })


def emit_budget_bypassed_trace() -> None:
    """
    Emit trace event when budgets are bypassed in experiment mode.
    """
    if not is_experiment_mode():
        return
    
    Trace.event("retrieval.budget.bypassed", {
        "search_budget_disabled": True,
        "extract_budget_disabled": True,
    })


def emit_sources_broadcast_trace(
    global_sources_count: int,
    per_claim_sources_count_total: int,
    broadcasted_sources_count: int,
) -> None:
    """
    Emit trace event when sources are broadcasted to all claims.
    
    Args:
        global_sources_count: Number of global sources
        per_claim_sources_count_total: Total per-claim sources across all claims
        broadcasted_sources_count: Final broadcasted count per claim
    """
    if not is_experiment_mode():
        return
    
    Trace.event("retrieval.sources.broadcast", {
        "global_sources_count": global_sources_count,
        "per_claim_sources_count_total": per_claim_sources_count_total,
        "broadcasted_sources_count": broadcasted_sources_count,
    })


def emit_escalation_full_run_trace(max_passes: int = 4) -> None:
    """
    Emit trace event when full escalation ladder is forced.
    
    Args:
        max_passes: Maximum number of escalation passes
    """
    if not is_experiment_mode():
        return
    
    Trace.event("retrieval.escalation.full_run", {
        "max_passes": max_passes,
        "per_claim": True,
    })


def build_broadcasted_sources(
    global_sources: list[dict[str, Any]],
    by_claim_sources: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Build a unified source list for broadcasting to all claims.
    
    When experiment mode is enabled, all sources (global + per-claim) are
    merged into a single list that will be used for every claim.
    
    Args:
        global_sources: List of global sources
        by_claim_sources: Dict mapping claim_id to list of sources
        
    Returns:
        List of all unique sources (by URL) with minimal fields
    """
    if not is_experiment_mode():
        # In normal mode, return empty (caller should use per-claim logic)
        return []
    
    seen_urls: set[str] = set()
    all_sources: list[dict[str, Any]] = []
    
    # Add global sources first
    for src in global_sources:
        url = src.get("url") or src.get("link") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            # Keep only minimal fields to avoid memory bloat
            all_sources.append({
                "url": url,
                "title": src.get("title", ""),
                "snippet": src.get("snippet") or src.get("content", ""),
                "content": src.get("content", ""),
                "score": src.get("score") or src.get("provider_score", 0.0),
            })
    
    # Add per-claim sources
    per_claim_count = 0
    for claim_id, sources in by_claim_sources.items():
        for src in sources:
            per_claim_count += 1
            url = src.get("url") or src.get("link") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_sources.append({
                    "url": url,
                    "title": src.get("title", ""),
                    "snippet": src.get("snippet") or src.get("content", ""),
                    "content": src.get("content", ""),
                    "score": src.get("score") or src.get("provider_score", 0.0),
                    # Preserve original claim_id in trace only
                    "_original_claim_id": str(claim_id),
                })
    
    # Emit trace
    emit_sources_broadcast_trace(
        global_sources_count=len(global_sources),
        per_claim_sources_count_total=per_claim_count,
        broadcasted_sources_count=len(all_sources),
    )
    
    return all_sources
