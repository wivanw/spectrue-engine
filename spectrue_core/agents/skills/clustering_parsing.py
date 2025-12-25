import hashlib
import json
import logging
from typing import Iterable, Union

from spectrue_core.schema import ClaimUnit
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.evidence_pack import SearchResult

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


def get_source_text_for_llm(source: dict, *, max_len: int = 350) -> tuple[str, bool, list[str]]:
    """
    M103: Return canonical text for LLM and metadata about source fields.

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

    content = (source.get("content") or source.get("extracted_content") or "").strip()
    if content:
        fields_present.append("content")

    # Priority: quote > snippet > content
    if quote:
        return quote[:max_len], True, fields_present

    if snippet:
        return snippet[:max_len], False, fields_present

    return content[:max_len], False, fields_present




def build_claims_lite(claims: list[Union[ClaimUnit, dict]]) -> list[dict]:
    """
    M103/M105: Build lightweight claim dicts for stance clustering LLM.
    
    Includes search_query to help LLM match sources to claims.
    Claims have queries in both search_queries and query_candidates fields.
    """
    claims_lite: list[dict] = []
    for c in (claims or []):
        if isinstance(c, ClaimUnit):
            assertions_lite = [{"key": a.key, "value": str(a.value)[:50]} for a in c.assertions]
            # M105: Get search query from search_queries or query_candidates
            search_queries = getattr(c, "search_queries", None) or getattr(c, "query_candidates", None) or []
            search_query = search_queries[0] if search_queries else (c.normalized_text or c.text)[:100]
            claims_lite.append(
                {
                    "id": c.id,
                    "text": c.normalized_text or c.text,
                    "assertions": assertions_lite,
                    "search_query": search_query,
                }
            )
        else:
            # M105: Get search query from search_queries or query_candidates
            search_queries = c.get("search_queries") or c.get("query_candidates") or []
            search_query = search_queries[0] if search_queries else c.get("text", "")[:100]
            claims_lite.append({
                "id": c.get("id"),
                "text": c.get("text"),
                "assertions": [],
                "search_query": search_query,
            })
    return claims_lite


def build_sources_lite(search_results: list[dict]) -> tuple[list[dict], set[int]]:
    """
    M103: Build lightweight source dicts for LLM with quote-carrying contract.

    Uses get_source_text_for_llm() for canonical text extraction.
    Adds has_quote and fields_present for contract audit.
    """
    results_lite: list[dict] = []
    unreadable_indices: set[int] = set()

    for i, r in enumerate(search_results or []):
        status_hint = ""
        if r.get("content_status") == "unavailable":
            status_hint = "[CONTENT UNAVAILABLE - JUDGE BY SNIPPET/TITLE]"

        # M103: Use canonical text extraction with quote priority
        source_text, has_quote, fields_present = get_source_text_for_llm(r, max_len=350)

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

        results_lite.append(
            {
                "index": i,
                "domain": r.get("domain") or r.get("url"),
                "title": r.get("title", ""),
                "text": f"{status_hint} {source_text}".strip(),
                # M103: Quote contract audit fields
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
        content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
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
        "kept_scored": 0,
        "kept_context": 0,
        "context_fallback": 0,
    }

    matrix_map = {m.get("source_index"): m for m in (matrix or []) if isinstance(m, dict)}
    valid_claim_ids = {c.get("id") for c in (claims_lite or [])}

    mapped_count = len(matrix_map)
    input_count = len(search_results)

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
            stats["context_fallback"] += 1
            clustered_results.append(_context_fallback_result(r))
            continue

        cid = match.get("claim_id")
        akey = match.get("assertion_key")
        stance = (match.get("stance") or "IRRELEVANT").upper()
        relevance = float(match.get("relevance", 0.0) or 0.0)
        quote = match.get("quote")

        if i in unreadable_indices:
            relevance = min(relevance, 0.1)
            stats["dropped_unreadable"] = stats.get("dropped_unreadable", 0) + 1

        if not cid or cid not in valid_claim_ids:
            cid = None
            stance = "CONTEXT"
            relevance = 0.0
            akey = None
            if not quote:
                quote = r.get("snippet", "")

        if stance not in {"CONTEXT", "IRRELEVANT"} and relevance < 0.4:
            stats["dropped_irrelevant"] += 1
            continue

        if stance not in valid_scoring_stances:
            if stance == "MENTION":
                stats["dropped_mention"] += 1
            else:
                stats["dropped_irrelevant"] += 1
            continue

        if stance not in {"CONTEXT", "IRRELEVANT"} and (not quote or not str(quote).strip()):
            stats["dropped_no_quote"] += 1
            continue

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
                content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
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
    logger.info("[Clustering] Matrix stats: %s", stats)

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
