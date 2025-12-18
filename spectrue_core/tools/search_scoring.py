# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Search Scoring Module
=====================
Provides relevance scoring, filtering, and ranking for search results.
Extracted from search_tool.py for maintainability (M61 refactoring).
"""

import datetime
import logging
import re
from urllib.parse import urlparse

from spectrue_core.verification.trusted_sources import ALL_TRUSTED_DOMAINS as TRUSTED_DOMAINS

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

# M61: Expanded archive domain detection
ARCHIVE_DOMAINS: set[str] = {
    # UN system
    "legal.un.org", "docs.un.org", "press.un.org", "undocs.org",
    "digitallibrary.un.org", "treaties.un.org",
    # Other IGOs
    "oecd.org", "oecd-ilibrary.org",
    "data.worldbank.org", "documents.worldbank.org",
    "imf.org",  # Main IMF often returns reports
    # Historical archives
    "history.state.gov", "archives.gov",
}

# URL path patterns that indicate archive/document content
ARCHIVE_URL_PATTERNS: set[str] = {"/record/", "/publications/", "/documents/", "/files/", "/-/media/"}

# Keywords that MUST be present to redeem an archive/PDF document
MUST_HAVE_KEYWORDS: set[str] = {
    "trump", "biden", "zelensky", "putin", "breaking", "news",
    "2024", "2025", "announced", "confirmed", "statement"
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

def relevance_score(
    query: str,
    title: str,
    content: str,
    url: str,
    *,
    tavily_score: float | None = None,
) -> float:
    """
    Lightweight lexical relevance scoring to drop obviously off-topic results.
    Keeps behavior deterministic and cheap (no LLM calls).
    
    Args:
        query: Search query string
        title: Result title
        content: Result content/snippet
        url: Result URL
        tavily_score: Optional Tavily-provided relevance score
    
    Returns:
        Relevance score between 0.0 and 1.0
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    tokens = [t for t in query_tokens if (len(t) >= 3 or t.isdigit()) and t not in STOP_WORDS]
    if not tokens:
        tokens = query_tokens[:8]

    title_tokens = set(tokenize(title))
    content_tokens = set(tokenize(content))

    hits_title = sum(1 for t in tokens if t in title_tokens)
    hits_content = sum(1 for t in tokens if t in content_tokens)
    hit_ratio = (hits_title * 2 + hits_content) / max(1, len(tokens))

    # Prefer sources that mention key numerals from query (dates, amounts)
    q_nums = {t for t in tokens if t.isdigit()}

    # Penalize extremely short snippets (often random/empty)
    length_penalty = 0.0
    if content and len(content) < 80:
        length_penalty = 0.08

    # Small bonus for trusted domains (stability)
    trusted_bonus = 0.0
    try:
        host = urlparse(url).netloc
        if is_trusted_host(host):
            trusted_bonus = 0.10
    except Exception:
        pass

    lexical_score = max(0.0, min(1.0, hit_ratio))

    # Start with lexical score
    final_score = lexical_score

    if tavily_score is not None:
        try:
            ts = float(tavily_score)
            # M46/M47: Hardened gate.
            # If keywords don't match (lexical < 0.15), we distrust Tavily
            # UNLESS it's extremely confident (ts >= 0.90), which suggests strong semantic match.
            if lexical_score < 0.15:
                if ts >= 0.90:
                    final_score = max(lexical_score, ts * 0.70)  # Reduced trust in pure semantic match
                else:
                    final_score = lexical_score * 0.5  # Heavy penalty
            else:
                # Blend: allow TS to boost, but cap the gain to avoid Tavily hallucination dominance.
                # M47: Fix "Tavily King" problem.
                blended = max(lexical_score, ts)
                cap = lexical_score + 0.35
                final_score = min(blended, cap)
        except Exception:
            pass

    final_score += trusted_bonus
    final_score -= length_penalty

    # M47: Must-have tokens penalty (Numbers)
    # If query has numbers (10000, 2026) but content doesn't, it's likely irrelevant.
    if q_nums:
        c_nums = {t for t in content_tokens if t.isdigit()}
        if not (q_nums & c_nums):
            final_score *= 0.5  # Heavy penalty

    # M36: Hard penalize specific forum patterns in known domains
    low_url = (url or "").lower()
    if "forum.nasaspaceflight.com" in low_url:
        final_score = 0.0

    # M61: Date freshness signal - boost recent content, penalize old archives
    freshness_adj = 0.0
    try:
        # Extract year from URL path (common patterns: /2025/12/, /2024-12-18/, /news/2025/)
        current_year = datetime.datetime.now().year
        year_match = re.search(r'/(\d{4})[-/]', url)
        if year_match:
            url_year = int(year_match.group(1))
            if url_year == current_year:
                freshness_adj = 0.08  # Boost current year
            elif url_year == current_year - 1:
                freshness_adj = 0.02  # Small boost for last year
            elif url_year < current_year - 2:
                freshness_adj = -0.12  # Penalize old content (>2 years)
    except Exception:
        pass
    final_score += freshness_adj

    return max(0.0, min(1.0, final_score))


# =============================================================================
# Rank and Filter
# =============================================================================

def rank_and_filter(query: str, results: list[dict]) -> list[dict]:
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
    scored: list[dict] = []
    for obj in results:
        s = relevance_score(
            query,
            obj.get("title", ""),
            obj.get("content", ""),
            obj.get("url", ""),
            tavily_score=obj.get("score"),
        )
        item = dict(obj)
        item["relevance_score"] = s
        scored.append(item)

    scored.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    # Keep best results; drop clearly off-topic tail.
    # M60: Advanced Noise Filtering to remove "poisonous" archives (UN, legal PDFs)
    kept = []
    discarded_reasons = []

    for r in scored:
        score = r.get("relevance_score", 0.0)
        url = r.get("url", "").lower()
        domain = urlparse(url).netloc
        content_lower = (r.get("content") or "").lower()

        # Check for generic PDF/Archive junk
        is_noise_domain = domain in ARCHIVE_DOMAINS or any(d in domain for d in ARCHIVE_DOMAINS)
        is_archive_url = any(p in url for p in ARCHIVE_URL_PATTERNS)
        is_pdf = ".pdf" in url

        if is_noise_domain or is_archive_url or is_pdf:
            # Only keep if strongly relevant to current news events
            has_news_keyword = any(k in content_lower for k in MUST_HAVE_KEYWORDS)
            if not has_news_keyword:
                discarded_reasons.append(f"{r.get('url')} (archive_garbage)")
                continue

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
        logger.info("[Search] All results below 0.15, keeping top 2 borderline results as fallback")
        kept = scored[:2]

    # M54: Enforce domain diversity
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
