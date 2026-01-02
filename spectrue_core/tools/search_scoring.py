# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Search Scoring Module
=====================
Provides relevance scoring, filtering, and ranking for search results.
Extracted from search_tool.py for maintainability (refactoring).
"""

import datetime
import logging
import re
from urllib.parse import urlparse

from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.calibration.calibration_models import logistic_score
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry
from spectrue_core.verification.search.trusted_sources import ALL_TRUSTED_DOMAINS as TRUSTED_DOMAINS

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Stopwords for multiple languages (used in tokenization filtering)
STOP_WORDS: set[str] = {
    # EN
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    # UA/RU common
    "і", "й", "та", "але", "або", "що", "це", "як", "в", "у", "на", "до", "з", "із", "за",
    "и", "й", "да", "но", "или", "что", "это", "как", "в", "на", "до", "с", "из", "за",
    # DE
    "der", "die", "das", "ein", "eine", "und", "oder", "zu", "von", "mit", "für", "im", "in", "am",
    "ist", "sind", "war", "waren", "sein", "wurde", "werden", "als", "auch", "auf", "bei", "nicht",
    # ES
    "el", "la", "los", "las", "un", "una", "y", "o", "de", "del", "en", "con", "por", "para", "como",
    "es", "son", "fue", "fueron", "ser", "al", "a", "que", "no", "se", "su", "sus",
    # FR
    "le", "la", "les", "un", "une", "et", "ou", "de", "des", "du", "en", "dans", "avec", "par", "pour",
    "est", "sont", "été", "etre", "être", "au", "aux", "ce", "cet", "cette", "ces", "que", "qui", "ne", "pas",
}





# Safety net: Do not trust known public suffixes during parent walk
FORBIDDEN_SUFFIXES: set[str] = {"co.uk", "com.au", "co.jp", "com.br", "co.id", "co.in", "com.cn", "org.uk"}


# =============================================================================
# Tokenization
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for relevance scoring.
    Handles Latin, Cyrillic, and CJK languages.
    """
    if not text:
        return []
    lower = text.lower()

    # For CJK languages (zh/ja), whitespace tokenization is weak; use character bigrams.
    def contains_cjk(s: str) -> bool:
        # Han + Hiragana + Katakana + Hangul
        return re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", s) is not None

    tokens: list[str] = []

    # Unicode-friendly word extraction (supports Cyrillic/Latin, etc.)
    tokens.extend(re.findall(r"[\w']+", lower))

    if contains_cjk(lower):
        cjk_only = re.sub(r"[^\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", "", lower)
        # Bigrams give a cheap signal for overlap without a full tokenizer.
        tokens.extend([cjk_only[i:i+2] for i in range(max(0, len(cjk_only) - 1))])

    return [t for t in tokens if t]


# =============================================================================
# Trust Check
# =============================================================================

def is_trusted_host(host: str) -> bool:
    """Centralized trusted host check with subdomain walking."""
    if not host:
        return False
    h = host.lower()
    if h.startswith("www."):
        h = h[4:]

    # Exact match
    if h in TRUSTED_DOMAINS:
        return True

    # Parent walk
    parts = h.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in FORBIDDEN_SUFFIXES:
            continue
        if candidate in TRUSTED_DOMAINS:
            return True
    return False


# =============================================================================
# Relevance Scoring
# =============================================================================

def _extract_recent_year(text: str) -> int | None:
    if not text:
        return None
    years = []
    for match in re.findall(r"\b(20\d{2})\b", text):
        try:
            year = int(match)
        except ValueError:
            continue
        years.append(year)
    if not years:
        return None
    now_year = datetime.datetime.now().year
    candidates = [y for y in years if 1990 <= y <= now_year + 1]
    if not candidates:
        return None
    return max(candidates)


def _year_freshness(url: str, title: str) -> float:
    year = _extract_recent_year(f"{url} {title}")
    if not year:
        return 0.0
    now_year = datetime.datetime.now().year
    delta = max(0, now_year - year)
    return max(0.0, min(1.0, 1.0 - (delta / 10.0)))


def relevance_score(
    query: str,
    title: str,
    content: str,
    url: str,
    *,
    tavily_score: float | None = None,
    runtime_config: EngineRuntimeConfig | None = None,
    trace: bool = False,
) -> float | tuple[float, dict]:
    """
    Lightweight lexical relevance scoring to drop obviously off-topic results.
    Keeps behavior deterministic and cheap (no LLM calls).
    
    Args:
        query: Search query string
        title: Result title
        content: Result content/snippet
        url: Result URL
        tavily_score: Optional Tavily-provided relevance score
        runtime_config: Optional engine runtime config for calibration models
        trace: Whether to emit trace payloads (returns score + trace)
    
    Returns:
        Relevance score between 0.0 and 1.0 (or score + trace)
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        if trace:
            trace_payload = {
                "model": "search_relevance",
                "version": "empty_query",
                "features": {},
                "score": 0.0,
                "fallback_used": True,
            }
            Trace.event("search.relevance.scored", trace_payload)
            return 0.0, trace_payload
        return 0.0

    tokens = [t for t in query_tokens if (len(t) >= 3 or t.isdigit()) and t not in STOP_WORDS]
    if not tokens:
        tokens = query_tokens[:8]

    title_tokens = set(tokenize(title))
    content_tokens = set(tokenize(content))

    hits_title = sum(1 for t in tokens if t in title_tokens)
    hits_content = sum(1 for t in tokens if t in content_tokens)
    hit_ratio = (hits_title * 2 + hits_content) / max(1, len(tokens))

    # Small bonus for trusted domains (stability)
    trusted_domain = 0.0
    try:
        host = urlparse(url).netloc
        if is_trusted_host(host):
            trusted_domain = 1.0
    except Exception:
        pass

    lexical_score = max(0.0, min(1.0, hit_ratio))
    snippet_len_norm = max(0.0, min(1.0, len(content or "") / 200.0))
    provider_score = 0.0
    if tavily_score is not None:
        try:
            provider_score = max(0.0, min(1.0, float(tavily_score)))
        except (TypeError, ValueError):
            provider_score = 0.0
    url_year_freshness = _year_freshness(url, title)

    features = {
        "lexical_score": lexical_score,
        "provider_score": provider_score,
        "snippet_len_norm": snippet_len_norm,
        "trusted_domain": trusted_domain,
        "url_year_freshness": url_year_freshness,
    }

    registry = CalibrationRegistry.from_runtime(runtime_config)
    model = registry.get_model("search_relevance")
    if model:
        final_score, trace_payload = model.score(features)
    else:
        policy = registry.policy.search_relevance
        raw, final_score = logistic_score(
            features,
            policy.fallback_weights or policy.weights,
            bias=policy.fallback_bias or policy.bias,
        )
        trace_payload = {
            "model": "search_relevance",
            "version": policy.version,
            "features": features,
            "weights": policy.fallback_weights or policy.weights,
            "bias": policy.fallback_bias or policy.bias,
            "raw_score": raw,
            "score": final_score,
            "fallback_used": True,
        }

    if trace:
        trace_payload["score"] = float(final_score)
        Trace.event("search.relevance.scored", trace_payload)
        return float(final_score), trace_payload
    return float(final_score)


# =============================================================================
# Rank and Filter
# =============================================================================

def rank_and_filter(
    query: str,
    results: list[dict],
    *,
    runtime_config: EngineRuntimeConfig | None = None,
) -> list[dict]:
    """
    Score, filter, and rank search results.
    
    - Computes relevance scores for each result
    - Filters out archive/PDF garbage
    - Enforces domain diversity
    - Returns top 10 results
    
    Args:
        query: Search query string
        results: List of result dicts with title, url, content, score
    
    Returns:
        Filtered and ranked list of results (max 10)
    """
    def _is_archive_like(url: str) -> bool:
        u = (url or "").lower()
        if not u:
            return False
        if u.endswith(".pdf"):
            return True
        if "/publications/" in u:
            return True
        if "digitallibrary.un.org/record/" in u:
            return True
        if "docs.un.org/record/" in u:
            return True
        return False

    def _has_news_signal(title: str, content: str, url: str) -> bool:
        blob = f"{title} {content} {url}".lower()
        if any(k in blob for k in ("breaking", "news", "update", "today", "current")):
            return True
        try:
            now_year = datetime.datetime.now().year
            years = {int(y) for y in re.findall(r"\\b(20\\d{2})\\b", blob)}
            return any(y >= now_year for y in years)
        except Exception:
            return False

    scored: list[dict] = []
    for obj in results:
        url = obj.get("url", "") or ""
        title = obj.get("title", "") or ""
        content = obj.get("content", "") or ""

        # T5: Filter archive/PDF-ish sources unless they look like fresh news.
        if _is_archive_like(url) and not _has_news_signal(title, content, url):
            continue

        score_result = relevance_score(
            query,
            title,
            content,
            url,
            tavily_score=obj.get("score"),
            runtime_config=runtime_config,
            trace=True,
        )
        if isinstance(score_result, tuple):
            s, trace_payload = score_result
        else:
            s, trace_payload = score_result, None
        item = dict(obj)
        item["relevance_score"] = s
        if trace_payload:
            item["relevance_trace"] = trace_payload
        scored.append(item)

    scored.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    # Log input stats for debugging
    if scored:
        tavily_scores = [r.get("score", 0) or 0 for r in scored]
        our_scores = [r.get("relevance_score", 0) for r in scored]
        logger.debug(
            "[Search] Input: %d results, tavily_max=%.2f, our_max=%.2f, query=%s",
            len(scored), max(tavily_scores), max(our_scores), query[:50]
        )

    # Keep best results; drop clearly off-topic tail.
    kept = []
    discarded_reasons = []

    for r in scored:
        score = r.get("relevance_score", 0.0)

        if score >= 0.15:
            kept.append(r)
        elif score >= 0.05 and not kept:
            # Fallback for borderline items
            kept.append(r)
        else:
            discarded_reasons.append(f"{r.get('url')} (low_score={score:.2f})")

    if discarded_reasons:
        logger.debug("[Search] Discarded %d results: %s", len(discarded_reasons), "; ".join(discarded_reasons[:5]))
    if not kept and scored:
        # Last resort fallback to avoid empty result if we had candidates
        logger.debug("[Search] All results below 0.15, keeping top 2 borderline results as fallback")
        kept = scored[:2]

    # Log output stats
    logger.debug("[Search] Output: %d kept, %d discarded", len(kept), len(discarded_reasons))

    # Enforce domain diversity
    # Select best result per domain first, then fill rest
    selected: list[dict] = []
    seen_domains: set[str] = set()
    backlog: list[dict] = []

    for r in kept:
        try:
            domain = urlparse(r.get("url", "")).netloc.lower()
            # strip www.
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = "unknown"

        if domain not in seen_domains:
            selected.append(r)
            seen_domains.add(domain)
        else:
            # Add to backlog but preserve order (since it was sorted by score)
            backlog.append(r)

    # If we have space left, fill with best from backlog (duplicates)
    target_count = 10
    if len(selected) < target_count:
        needed = target_count - len(selected)
        # Re-sort backlog by score just in case, though it should be sorted
        backlog.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        selected.extend(backlog[:needed])

    return selected[:target_count]