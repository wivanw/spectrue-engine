# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
from __future__ import annotations

import logging
from typing import Any

from spectrue_core.utils.text_processing import normalize_search_query

logger = logging.getLogger(__name__)


def get_query_by_role(claim: dict, role: str) -> str | None:
    """
    M64: Extract query with specific role from claim's query_candidates.

    Args:
        claim: Claim dict with query_candidates and/or search_queries
        role: Query role ("CORE", "NUMERIC", "ATTRIBUTION", "LOCAL")

    Returns:
        Query text or None if not found
    """
    for candidate in claim.get("query_candidates", []):
        if candidate.get("role") == role:
            return candidate.get("text", "")

    legacy = claim.get("search_queries", [])
    role_index = {"CORE": 0, "NUMERIC": 1, "ATTRIBUTION": 2, "LOCAL": 0}
    idx = role_index.get(role, 0)
    if idx < len(legacy):
        return legacy[idx]
    if role == "CORE" and legacy:
        return legacy[0]
    return None


def normalize_and_sanitize(query: str) -> str | None:
    """
    M64: Normalize query.

    Note: Strict gambling keywords removal is deprecated (M64).
    We rely on LLM constraints and Tavily 'topic="news"' mode
    to prevent gambling/spam results instead of hardcoded stoplists.
    """
    if not query:
        return None
    normalized = normalize_search_query(query)
    return normalized or None


def is_fuzzy_duplicate(
    query: str,
    existing: list[str],
    *,
    threshold: float = 0.9,
    log: logging.Logger = logger,
) -> bool:
    """
    M64: Check if query is >threshold similar to any existing query.

    Uses Jaccard similarity on word sets.
    """
    query_words = set(query.lower().split())
    if not query_words:
        return True

    for existing_query in existing:
        existing_words = set(existing_query.lower().split())
        if not existing_words:
            continue

        intersection = len(query_words & existing_words)
        union = len(query_words | existing_words)
        similarity = intersection / union if union > 0 else 0

        if similarity >= threshold:
            log.debug(
                "[M64] Fuzzy duplicate (%.0f%%): '%s' ≈ '%s'",
                similarity * 100,
                query[:30],
                existing_query[:30],
            )
            return True

    return False


def select_diverse_queries(
    claims: list,
    *,
    max_queries: int = 3,
    fact_fallback: str = "",
    log: logging.Logger = logger,
) -> list[str]:
    """
    M64: Topic-Aware Round-Robin Query Selection ("Coverage Engine").

    Ensures every topic_key gets at least 1 query before any topic gets
    a 2nd query. This solves the "Digest Problem" where multi-topic
    articles had all queries spent on the first topic only.
    """
    from collections import defaultdict

    if not claims:
        return [normalize_search_query(fact_fallback[:200])] if fact_fallback else []

    MIN_WORTHINESS = 0.4
    eligible = [
        c
        for c in claims
        if c.get("type") != "sidefact"
        and c.get("check_worthiness", c.get("importance", 0.5)) >= MIN_WORTHINESS
    ]

    if not eligible:
        eligible = sorted(
            [c for c in claims if c.get("type") != "sidefact"],
            key=lambda c: c.get("importance", 0),
            reverse=True,
        )[:1]
        if not eligible:
            eligible = claims[:1]
        log.debug("[M64] All claims below threshold or sidefacts, using highest importance")

    groups: dict[str, list] = defaultdict(list)
    for c in eligible:
        key = c.get("cluster_id") or c.get("topic_key") or c.get("topic_group", "Other")
        groups[key].append(c)

    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: max((c.get("importance", 0) for c in groups[k]), default=0),
        reverse=True,
    )

    for key in groups:
        groups[key].sort(key=lambda c: (-c.get("importance", 0.0), c.get("text", "")))

    selected: list[str] = []
    covered_topics: set[str] = set()

    for key in sorted_keys:
        if len(selected) >= max_queries:
            break

        top_claim = groups[key][0]
        core_query = get_query_by_role(top_claim, "CORE")

        if core_query:
            normalized = normalize_and_sanitize(core_query)
            if normalized and not is_fuzzy_duplicate(normalized, selected, threshold=0.9, log=log):
                selected.append(normalized)
                covered_topics.add(key)
                log.debug("[M64] Pass 1 (Coverage): topic=%s, query=%s", key, normalized[:50])

    if len(selected) < max_queries:
        for key in sorted_keys:
            if len(selected) >= max_queries:
                break
            if key not in covered_topics:
                continue

            top_claim = groups[key][0]
            for role in ["NUMERIC", "ATTRIBUTION"]:
                query = get_query_by_role(top_claim, role)
                if query:
                    normalized = normalize_and_sanitize(query)
                    if normalized and not is_fuzzy_duplicate(normalized, selected, threshold=0.9, log=log):
                        selected.append(normalized)
                        log.debug(
                            "[M64] Pass 2 (Depth): topic=%s, role=%s, query=%s",
                            key,
                            role,
                            normalized[:50],
                        )
                        break

    if len(selected) < max_queries:
        for key in sorted_keys:
            if len(selected) >= max_queries:
                break
            for claim in groups[key]:
                if len(selected) >= max_queries:
                    break
                for candidate in claim.get("query_candidates", []):
                    query = candidate.get("text", "")
                    normalized = normalize_and_sanitize(query)
                    if normalized and not is_fuzzy_duplicate(normalized, selected, threshold=0.9, log=log):
                        selected.append(normalized)
                        log.debug("[M64] Pass 3 (Fill): topic=%s, query=%s", key, normalized[:50])
                        if len(selected) >= max_queries:
                            break
                for query in claim.get("search_queries", []):
                    normalized = normalize_and_sanitize(query)
                    if normalized and not is_fuzzy_duplicate(normalized, selected, threshold=0.9, log=log):
                        selected.append(normalized)
                        log.debug("[M64] Pass 3 (Fill/Legacy): topic=%s, query=%s", key, normalized[:50])
                        if len(selected) >= max_queries:
                            break

    if not selected and fact_fallback:
        selected = [normalize_search_query(fact_fallback[:200])]

    log.debug(
        "[M64] Query selection: %d eligible claims, %d topic_keys -> %d queries. Topics covered: %s",
        len(eligible),
        len(groups),
        len(selected),
        list(covered_topics)[:5],
    )

    return selected[:max_queries]


def resolve_budgeted_max_queries(claims: list, *, default_max: int = 3) -> int:
    """
    Apply per-claim policy caps to the run-level query budget.
    """
    if not claims:
        return default_max

    modes = {c.get("policy_mode") for c in claims if isinstance(c, dict)}
    if modes and modes.issubset({"SKIP"}):
        return 0
    if modes and modes.issubset({"SKIP", "CHEAP"}):
        return 1

    budget_caps: list[int] = []
    for claim in claims:
        budget = claim.get("budget_allocation") if isinstance(claim, dict) else None
        if isinstance(budget, dict):
            cap = budget.get("max_queries")
            if isinstance(cap, int) and cap > 0:
                budget_caps.append(cap)

    if budget_caps:
        return max(1, min(default_max, min(budget_caps)))

    return default_max


def build_assertion_query(unit: Any, assertion: Any) -> str | None:
    """
    M70: Build search query for a specific assertion.

    Query structure: "{subject} {assertion.value} {context}" (heuristics unchanged).
    """
    from spectrue_core.schema import Assertion, Dimension

    if not isinstance(assertion, Assertion):
        return None

    if assertion.dimension == Dimension.CONTEXT:
        return None

    parts: list[str] = []

    if unit.subject:
        parts.append(unit.subject)
    if unit.object:
        parts.append(unit.object)

    if assertion.value:
        value_str = str(assertion.value)
        if len(value_str) < 50:
            parts.append(value_str)

    key = assertion.key
    if "location" in key:
        parts.append("location official")
    elif "time" in key or "date" in key:
        parts.append("date confirmed")
    elif "quote" in key or "attribution" in key:
        parts.append("said statement")
    elif "numeric" in key or "value" in key:
        parts.append("official data")
    else:
        parts.append("verified")

    return " ".join(parts) if parts else None


def select_queries_from_claim_units(
    claim_units: list,
    *,
    max_queries: int = 3,
    fact_fallback: str = "",
    log: logging.Logger = logger,
) -> list[str]:
    """
    M70: Extract verification queries for FACT assertions in structured ClaimUnits.
    """
    from spectrue_core.schema import ClaimUnit

    if not claim_units:
        return [normalize_search_query(fact_fallback[:200])] if fact_fallback else []

    queries: list[str] = []

    for unit in claim_units:
        if not isinstance(unit, ClaimUnit):
            continue

        fact_assertions = unit.get_fact_assertions()
        for assertion in fact_assertions:
            if len(queries) >= max_queries:
                break

            query = build_assertion_query(unit, assertion)
            if not query:
                continue

            normalized = normalize_and_sanitize(query)
            if normalized and not is_fuzzy_duplicate(normalized, queries, threshold=0.9, log=log):
                queries.append(normalized)
                log.debug("[M70] Assertion query: key=%s, query=%s", assertion.key, normalized[:50])

    if not queries:
        for unit in claim_units:
            if isinstance(unit, ClaimUnit) and unit.text:
                return [normalize_search_query(unit.text[:200])]
        if fact_fallback:
            return [normalize_search_query(fact_fallback[:200])]

    log.debug("[M70] Query selection: %d claims, %d FACT queries generated", len(claim_units), len(queries))
    return queries[:max_queries]


def get_claim_units_for_evidence_mapping(claim_units: list, sources: list[dict]) -> dict[str, list[str]]:
    """
    M70: Map claim_id -> assertion_keys (FACT) for targeted verification.

    `sources` is accepted for back-compat with current call sites.
    """
    from spectrue_core.schema import ClaimUnit, Dimension

    mapping: dict[str, list[str]] = {}

    for unit in claim_units:
        if not isinstance(unit, ClaimUnit):
            continue

        fact_keys = [a.key for a in unit.assertions if a.dimension == Dimension.FACT]
        if fact_keys:
            mapping[unit.id] = fact_keys

    return mapping
