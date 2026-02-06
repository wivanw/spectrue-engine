# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import hashlib
import json
import logging
from typing import Iterable, Union

from spectrue_core.schema import ClaimUnit
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.evidence.evidence_pack import SearchResult
from spectrue_core.verification.search.source_utils import has_evidence_chunk
from spectrue_core.verification.scoring.stance_posterior import (
    StanceFeatures,
    compute_stance_posterior,
    source_prior_from_tier,
)

logger = logging.getLogger(__name__)

UNREADABLE_MARKERS = [
    "access denied",
    "403 forbidden",
    "404 not found",
    "javascript required",
    "please enable javascript",
    "cloudflare",
    "captcha",
    "robot check",
]


def get_source_text_for_llm(source: dict, *, max_len: int = 2500) -> tuple[str, bool, list[str]]:
    """
    Return canonical text for LLM and metadata about source fields.

    Priority: quote > snippet > content/extracted_content

    Args:
        source: Source dict from search results
        max_len: Maximum length of returned text

    Returns:
        Tuple of (text, has_quote_flag, fields_present)
    """
    fields_present: list[str] = []

    quote = (source.get("quote") or "").strip()
    if quote:
        fields_present.append("quote")

    snippet = (source.get("snippet") or "").strip()
    if snippet:
        fields_present.append("snippet")

    content = (
        source.get("content")
        or source.get("content_excerpt")
        or source.get("extracted_content")
        or ""
    ).strip()
    if content:
        fields_present.append("content")

    # Priority: quote > snippet > content
    if quote:
        return quote[:max_len], True, fields_present

    if snippet:
        return snippet[:max_len], False, fields_present

    if content:
        return content[:max_len], False, fields_present

    title = (source.get("title") or "").strip()
    if title:
        fields_present.append("title")
        return title[:max_len], False, fields_present

    url = (source.get("url") or source.get("link") or "").strip()
    if url:
        fields_present.append("url")
        return url[:max_len], False, fields_present

    return "", False, fields_present




def build_claims_lite(claims: list[Union[ClaimUnit, dict]]) -> list[dict]:
    """
    M103/Build lightweight claim dicts for stance clustering LLM.
    
    Includes search_query to help LLM match sources to claims.
    Claims have queries in both search_queries (strings) and query_candidates (dicts).
    """
    claims_lite: list[dict] = []
    for c in (claims or []):
        if isinstance(c, ClaimUnit):
            assertions_lite = [{"key": a.key, "value": str(a.value)[:50]} for a in c.assertions]
            search_query = _extract_search_query(c)
            claims_lite.append(
                {
                    "id": c.id,
                    "text": c.normalized_text or c.text,
                    "assertions": assertions_lite,
                    "search_query": search_query,
                }
            )
        else:
            search_query = _extract_search_query(c)
            claims_lite.append({
                "id": c.get("id"),
                "text": c.get("text"),
                "assertions": [],
                "search_query": search_query,
            })
    return claims_lite


def _extract_search_query(claim) -> str:
    """
    Extract best search query from claim.
    
    Priority:
    1. search_queries[0] (plain strings)
    2. query_candidates[0]["text"] (dicts with text key)
    3. Fallback to claim text[:100]
    """
    # Try search_queries first (list of strings)
    if hasattr(claim, "search_queries"):
        sq = claim.search_queries
    else:
        sq = claim.get("search_queries") if isinstance(claim, dict) else []

    if sq and len(sq) > 0:
        return sq[0] if isinstance(sq[0], str) else str(sq[0])

    # Try query_candidates (list of dicts with "text" key)
    if hasattr(claim, "query_candidates"):
        qc = claim.query_candidates
    else:
        qc = claim.get("query_candidates") if isinstance(claim, dict) else []

    if qc and len(qc) > 0:
        first = qc[0]
        if isinstance(first, dict):
            return first.get("text", "")[:100]
        return str(first)[:100]

    # Fallback to claim text (prefer original text over normalized)
    # INVARIANT: search queries should use original language, not normalized
    if hasattr(claim, "text"):
        return (claim.text or "")[:100]
    if isinstance(claim, dict):
        return claim.get("text", "")[:100]
    return ""


def build_sources_lite(search_results: list[dict]) -> tuple[list[dict], set[int]]:
    """
    Build lightweight source dicts for LLM with quote-carrying contract.

    Uses get_source_text_for_llm() for canonical text extraction.
    Adds has_quote and fields_present for contract audit.
    """
    results_lite: list[dict] = []
    unreadable_indices: set[int] = set()

    for i, r in enumerate(search_results or []):
        status_hint = ""
        if r.get("content_status") == "unavailable":
            status_hint = "[CONTENT UNAVAILABLE - JUDGE BY SNIPPET/TITLE]"

        # Use canonical text extraction with quote priority
        source_text, has_quote, fields_present = get_source_text_for_llm(r, max_len=2500)

        # Check for unreadable content markers
        is_unreadable = False
        if len(source_text) < 50:
            is_unreadable = True
            status_hint = "[UNREADABLE: Content too short]"

        content_lower = source_text.lower()
        for marker in UNREADABLE_MARKERS:
            if marker in content_lower:
                is_unreadable = True
                status_hint = f"[UNREADABLE: {marker.upper()}]"
                break

        if is_unreadable:
            unreadable_indices.add(i)

        quote_value = (r.get("quote") or "").strip()
        if quote_value:
            fields_present.append("quote_value")

        max_quote_len = 500
        results_lite.append(
            {
                "index": i,
                "domain": r.get("domain") or r.get("url"),
                "title": r.get("title", ""),
                "text": f"{status_hint} {source_text}".strip(),
                "quote": quote_value[:max_quote_len] if quote_value else "",
                # Quote contract audit fields
                "has_quote": has_quote,
                "fields_present": fields_present,
            }
        )

    return results_lite, unreadable_indices


def build_ev_mat_cache_key(*, claims_lite: list[dict], sources_lite: list[dict]) -> str:
    content_hash = hashlib.sha256(
        (json.dumps(claims_lite, sort_keys=True) + json.dumps(sources_lite, sort_keys=True)).encode()
    ).hexdigest()[:32]
    return f"ev_mat_v1_{content_hash}"


def _context_fallback_result(r: dict) -> SearchResult:
    return SearchResult(
        claim_id=None,  # type: ignore
        url=r.get("url") or r.get("link") or "",
        domain=r.get("domain"),
        title=r.get("title", ""),
        snippet=r.get("snippet", ""),
        content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:25000],
        published_at=r.get("published_date"),
        source_type=r.get("source_type", "unknown"),  # type: ignore
        stance="context",  # type: ignore
        relevance_score=0.0,
        quote_matches=[],
        key_snippet=r.get("snippet", ""),
        is_trusted=bool(r.get("is_trusted")),
        is_duplicate=False,
        duplicate_of=None,
        assertion_key=None,
        content_status=r.get("content_status", "available"),
        evidence_tier=r.get("evidence_tier"),
        evidence_role="mention_only",
        covers=[],
    )


def postprocess_evidence_matrix(
    *,
    search_results: list[dict],
    claims_lite: list[dict],
    matrix: list[dict],
    unreadable_indices: set[int],
    valid_scoring_stances: set[str],
) -> tuple[list[SearchResult], dict]:
    clustered_results: list[SearchResult] = []
    stats: dict = {
        "input_sources": len(search_results),
        "dropped_irrelevant": 0,
        "dropped_mention": 0,
        "dropped_bad_id": 0,
        "dropped_no_quote": 0,
        "dropped_missing": 0,
        "downgraded_low_relevance": 0,
        "context_promoted": 0,
        "kept_scored": 0,
        "kept_context": 0,
        "context_fallback": 0,
        "fallback_scored": 0,
        "quote_filled": 0,
    }

    matrix_map = {m.get("source_index"): m for m in (matrix or []) if isinstance(m, dict)}
    valid_claim_ids = {c.get("id") for c in (claims_lite or [])}
    mapped_count = len(matrix_map)
    input_count = len(search_results)

    Trace.event(
        "matrix.summary",
        {
            "input_sources": input_count,
            "mapped_rows": mapped_count,
            "claim_count": len(claims_lite),
        },
    )

    if mapped_count == 0 and input_count > 0:
        Trace.event(
            "matrix.failure",
            {
                "mode": "empty",
                "input_sources": input_count,
                "mapped_rows": 0,
                "claim_count": len(claims_lite),
            },
        )
        logger.warning("[Clustering] ⚠️ Matrix EMPTY: LLM returned 0 rows for %d sources", input_count)
    elif mapped_count < input_count:
        Trace.event(
            "matrix.failure",
            {
                "mode": "partial",
                "input_sources": input_count,
                "mapped_rows": mapped_count,
                "missing_sources": input_count - mapped_count,
            },
        )
        logger.warning("[Clustering] ⚠️ Matrix PARTIAL: %d/%d sources mapped", mapped_count, input_count)

    for i, r in enumerate(search_results):
        match = matrix_map.get(i)
        if not match:
            source_claim_id = r.get("claim_id")
            if source_claim_id and has_evidence_chunk(r):
                quote = r.get("quote") or r.get("snippet") or r.get("content") or ""
                clustered_results.append(
                    SearchResult(
                        claim_id=str(source_claim_id),  # type: ignore
                        url=r.get("url") or r.get("link") or "",
                        domain=r.get("domain"),
                        title=r.get("title", ""),
                        snippet=r.get("snippet", ""),
                        content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:25000],
                        published_at=r.get("published_date"),
                        source_type=r.get("source_type", "unknown"),  # type: ignore
                        stance="neutral",  # type: ignore
                        relevance_score=float(r.get("relevance_score", 0.0) or 0.0),
                        quote_matches=[quote] if quote else [],
                        key_snippet=quote if quote else r.get("snippet", ""),
                        is_trusted=bool(r.get("is_trusted")),
                        is_duplicate=False,
                        duplicate_of=None,
                        assertion_key=None,
                        content_status=r.get("content_status", "available"),
                        evidence_tier=r.get("evidence_tier"),
                    )
                )
                stats["fallback_scored"] += 1
                stats["kept_scored"] += 1
            else:
                stats["context_fallback"] += 1
                clustered_results.append(_context_fallback_result(r))
            continue

        source_claim_id = r.get("claim_id")
        cid = match.get("claim_id")
        akey = match.get("assertion_key")
        stance = (match.get("stance") or "IRRELEVANT").upper()
        relevance = float(match.get("relevance", 0.0) or 0.0)
        quote = match.get("quote")

        # --- Evidence metadata (typed, non-heuristic) ---
        evidence_role = match.get("evidence_role", "indirect")
        covers = match.get("covers", [])
        # Rule (deterministic, non-heuristic): quote present -> role is direct
        if quote and str(quote).strip():
            evidence_role = "direct"

        if i in unreadable_indices:
            relevance = min(relevance, 0.1)
            stats["dropped_unreadable"] = stats.get("dropped_unreadable", 0) + 1

        if source_claim_id and source_claim_id in valid_claim_ids:
            cid = str(source_claim_id)
        elif not cid or cid not in valid_claim_ids:
            source_claim_id = r.get("claim_id")
            if source_claim_id and has_evidence_chunk(r):
                cid = str(source_claim_id)
                stance = "NEUTRAL"
                if quote is None:
                    quote = r.get("quote") or r.get("snippet") or r.get("content") or ""
            else:
                cid = None
                stance = "CONTEXT"
                relevance = 0.0
                akey = None
                if not quote:
                    quote = r.get("snippet", "")

        # =========================================================================
        # BAYESIAN STANCE POSTERIOR (M113+)
        # Replaces hard rule: if relevance < 0.4 => NEUTRAL
        # Now uses soft P(S|features) calculation
        # =========================================================================

        # Prepare structural features for posterior calculation
        quote_for_features = quote or r.get("quote") or ""
        has_chunk = has_evidence_chunk(r)
        tier = r.get("evidence_tier")
        source_prior = source_prior_from_tier(tier)

        features = StanceFeatures(
            llm_stance=stance,
            llm_relevance=relevance if relevance > 0 else None,
            quote_present=bool(quote_for_features and str(quote_for_features).strip()),
            has_evidence_chunk=has_chunk,
            source_prior=source_prior,
        )

        posterior = compute_stance_posterior(features)

        # Decision: Use argmax stance if p_evidence > threshold
        # This is softer than hard rule: we trust LLM more when evidence signals are strong
        EVIDENCE_THRESHOLD = 0.35  # P(S ∈ {SUPPORT, REFUTE}) threshold

        if posterior.p_evidence >= EVIDENCE_THRESHOLD:
            # Strong evidence signal - use argmax (likely SUPPORT or REFUTE)
            stance = posterior.argmax_stance.upper()
            # Boost relevance proportionally to p_evidence
            relevance = max(relevance, posterior.p_evidence)
            stats["bayesian_kept"] = stats.get("bayesian_kept", 0) + 1
        elif stance in {"SUPPORT", "REFUTE", "MIXED"}:
            # LLM said evidence but posterior disagrees - soft downgrade
            if posterior.p_neutral > posterior.p_evidence:
                stance = "NEUTRAL"
                stats["bayesian_downgraded"] = stats.get("bayesian_downgraded", 0) + 1
            else:
                # Keep stance but note uncertainty
                stats["bayesian_uncertain"] = stats.get("bayesian_uncertain", 0) + 1
        elif stance in {"CONTEXT", "IRRELEVANT"} and cid and has_chunk:
            # Context promotion based on structural signals
            stance = "NEUTRAL"
            stats["context_promoted"] += 1
            if quote is None:
                quote = r.get("quote") or r.get("snippet") or r.get("content") or ""

        if stance not in valid_scoring_stances:
            if stance == "MENTION":
                stats["dropped_mention"] += 1
            else:
                stats["dropped_irrelevant"] += 1
            continue

        if stance in {"SUPPORT", "REFUTE", "MIXED"} and (not quote or not str(quote).strip()):
            fallback_quote = r.get("quote") or r.get("snippet") or r.get("content") or ""
            if fallback_quote:
                quote = fallback_quote
                stats["quote_filled"] += 1
            else:
                stats["dropped_no_quote"] += 1
                stance = "NEUTRAL"

        if stance in {"SUPPORT", "REFUTE", "MIXED"}:
            stats["kept_scored"] += 1
        else:
            stats["kept_context"] += 1

        clustered_results.append(
            SearchResult(
                claim_id=cid,  # type: ignore
                url=r.get("url") or r.get("link") or "",
                domain=r.get("domain"),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:25000],
                published_at=r.get("published_date"),
                source_type=r.get("source_type", "unknown"),  # type: ignore
                stance=stance.lower(),  # type: ignore
                relevance_score=relevance,
                quote_matches=[quote] if quote else [],
                key_snippet=quote,
                is_trusted=bool(r.get("is_trusted")),
                is_duplicate=False,
                duplicate_of=None,
                assertion_key=akey,
                content_status=r.get("content_status", "available"),
                evidence_tier=r.get("evidence_tier"),
                # Evidence metadata
                evidence_role=evidence_role,  # type: ignore
                covers=covers,
                # Bayesian posterior (M113+)
                p_support=posterior.p_support,
                p_refute=posterior.p_refute,
                p_neutral=posterior.p_neutral,
                p_evidence=posterior.p_evidence,
                posterior_entropy=posterior.entropy,
            )
        )

    context_count = sum(1 for r in clustered_results if r.get("stance") == "context")
    if clustered_results and context_count / len(clustered_results) > 0.7:
        Trace.event(
            "matrix.degraded",
            {
                "context_ratio": round(context_count / len(clustered_results), 2),
                "threshold": 0.7,
                "input_sources": input_count,
            },
        )
        logger.warning(
            "[Clustering] ⚠️ Matrix DEGRADED: %.0f%% sources are CONTEXT",
            (context_count / len(clustered_results)) * 100,
        )

    Trace.event("evidence.synthesis_stats", stats)

    # M113+: Bayesian posterior summary for calibration monitoring
    if clustered_results:
        effective_support = sum(r.get("p_support", 0) for r in clustered_results)
        effective_refute = sum(r.get("p_refute", 0) for r in clustered_results)
        effective_evidence = sum(r.get("p_evidence", 0) for r in clustered_results)
        mean_entropy = sum(r.get("posterior_entropy", 0) for r in clustered_results) / len(clustered_results)
        Trace.event("evidence.bayesian_summary", {
            "effective_support": round(effective_support, 3),
            "effective_refute": round(effective_refute, 3),
            "effective_evidence": round(effective_evidence, 3),
            "mean_entropy": round(mean_entropy, 3),
            "total_sources": len(clustered_results),
            "bayesian_kept": stats.get("bayesian_kept", 0),
            "bayesian_downgraded": stats.get("bayesian_downgraded", 0),
            "bayesian_uncertain": stats.get("bayesian_uncertain", 0),
        })

    logger.debug("[Clustering] Matrix stats: %s", stats)

    return clustered_results, stats


def exception_fallback_all_context(*, search_results: list[dict], error: Exception) -> list[SearchResult]:
    logger.warning("[Clustering] ⚠️ Evidence Matrix LLM failed: %s. Converting all to CONTEXT.", error)
    Trace.event(
        "matrix.failure",
        {
            "mode": "exception",
            "input_sources": len(search_results),
            "error": str(error)[:200],
        },
    )
    return [_context_fallback_result(r) for r in (search_results or [])]


def build_valid_claim_ids(claims_lite: Iterable[dict]) -> set[str]:
    return {c.get("id") for c in (claims_lite or []) if c.get("id")}
