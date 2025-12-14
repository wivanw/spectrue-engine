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

import os
import asyncio
import httpx
import logging
import re
from pathlib import Path
import diskcache
from typing import Tuple
from urllib.parse import urlparse
from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.runtime import is_local_run
from spectrue_core.utils.trace import Trace

# M29: Import from centralized trusted sources registry
from spectrue_core.verification.trusted_sources import ALL_TRUSTED_DOMAINS as TRUSTED_DOMAINS

logger = logging.getLogger(__name__)


class WebSearchTool:
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        api_key = config.tavily_api_key if config else os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            # Fallback to env var if config is missing (legacy)
            api_key = os.getenv("TAVILY_API_KEY")
            
        if not api_key:
            # Only raise if truly missing
            # In library mode, user might want to initialize tool but set key later?
            # But search won't work.
            pass 
            
        self.api_key = api_key

        self.client = httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "X-Client-Source": "spectrue",
            },
        )

        cache_dir = Path("data/cache/web")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_dir))
        self.ttl = 86400

        page_cache_dir = Path("data/cache/page_text")
        page_cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_cache = diskcache.Cache(str(page_cache_dir))
        self.page_ttl = 86400 * 7

        try:
            conc = int(os.getenv("TAVILY_CONCURRENCY", "8"))
        except Exception:
            conc = 8
        self._sem = asyncio.Semaphore(max(1, min(conc, 16)))
        self.last_cache_hit: bool = False
        self.last_search_meta: dict = {}

    def country_for_lang(self, lang: str | None) -> str | None:
        """Deprecated: do not infer `country` from language."""
        return None

    def _normalize_params(self, search_depth: str, max_results: int) -> Tuple[str, int]:
        depth = (search_depth or "basic").lower()
        if depth not in ("basic", "advanced"):
            depth = "basic"
        m = int(max_results) if max_results else 10  # M29: Increased to 10 for deeper context
        return depth, max(1, min(m, 15))

    def _is_valid_url(self, url: str) -> bool:
        try:
            if not url:
                return False
            u = urlparse(str(url).strip())
            if u.scheme not in ("http", "https"):
                return False
            host = (u.hostname or "").lower()
            if host in ("127.0.0.1", "localhost"):
                return False
            return True
        except Exception:
            return False

    def _clean_results(self, results: list[dict]) -> list[dict]:
        """Filter Tavily results: drop invalid/localhost/empty links, dedupe, and ensure title fallback."""
        cleaned: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for obj in (results or []):
            title = (obj.get("title") or "").strip()
            url = (obj.get("url") or "").strip()
            content = (obj.get("content") or "").strip()
            raw_content = (obj.get("raw_content") or "").strip()
            score = obj.get("score")  # Tavily may provide relevance score

            if not self._is_valid_url(url):
                continue

            # Fallback title to domain to avoid "Без заголовка"
            if not title:
                try:
                    title = urlparse(url).netloc or url
                except Exception:
                    title = url

            key = (title.lower(), url.lower())
            if key in seen:
                continue
            seen.add(key)

            use_raw = False
            chosen = content
            if raw_content and len(raw_content) > max(200, len(content)):
                use_raw = True
                chosen = raw_content
            chosen = re.sub(r"\s+", " ", chosen).strip()
            if len(chosen) > 2500:
                chosen = chosen[:2500]

            cleaned.append(
                {
                    "title": title,
                    "url": url,
                    "content": chosen,
                    "score": score,
                    "raw_content_used": use_raw,
                }
            )
        return cleaned

    def _tokenize(self, text: str) -> list[str]:
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

    def _relevance_score(self, query: str, title: str, content: str, url: str, *, tavily_score: float | None) -> float:
        """
        Lightweight lexical relevance scoring to drop obviously off-topic results.
        Keeps behavior deterministic and cheap (no LLM calls).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        stop = {
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
        tokens = [t for t in query_tokens if (len(t) >= 3 or t.isdigit()) and t not in stop]
        if not tokens:
            tokens = query_tokens[:8]

        title_tokens = set(self._tokenize(title))
        content_tokens = set(self._tokenize(content))

        hits_title = sum(1 for t in tokens if t in title_tokens)
        hits_content = sum(1 for t in tokens if t in content_tokens)
        hit_ratio = (hits_title * 2 + hits_content) / max(1, len(tokens))

        # Prefer sources that mention key numerals from query (dates, amounts)
        q_nums = {t for t in tokens if t.isdigit()}
        num_bonus = 0.0
        if q_nums:
            c_nums = {t for t in content_tokens if t.isdigit()}
            num_bonus = 0.15 if (q_nums & c_nums) else 0.0

        # Penalize extremely short snippets (often random/empty)
        length_penalty = 0.0
        if content and len(content) < 80:
            length_penalty = 0.08

        # Small bonus for trusted domains (stability)
        trusted_bonus = 0.0
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            if host in TRUSTED_DOMAINS:
                trusted_bonus = 0.10
        except Exception:
            pass

        base = max(0.0, min(1.0, hit_ratio))
        if tavily_score is not None:
            try:
                base = max(base, float(tavily_score))
            except Exception:
                pass

        return max(0.0, min(1.0, base + num_bonus + trusted_bonus - length_penalty))

    def _rank_and_filter(self, query: str, results: list[dict]) -> list[dict]:
        scored: list[dict] = []
        for obj in results:
            s = self._relevance_score(
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
        kept = [r for r in scored if r.get("relevance_score", 0.0) >= 0.12]
        if not kept:
            kept = scored[:6]
        return kept[:10]

    def _detect_topic(self, query: str) -> str:
        combined = (query or "").lower()

        finance_keywords = [
            "stock", "stocks", "market", "shares", "earnings", "ipo", "fund",
            "bitcoin", "btc", "ethereum", "eth", "crypto", "price", "trading",
            "usd", "eur", "uah", "inflation", "rate", "interest",
        ]
        news_keywords = [
            "today", "yesterday", "this week", "breaking", "latest", "new", "recent", "now",
            "сьогодні", "вчора", "щойно", "останні", "терміново", "зараз",
            "сегодня", "вчера", "только что", "срочно", "сейчас",
        ]
        science_keywords = [
            "comet", "asteroid", "nasa", "astronomy", "planet", "space",
            "research", "study", "scientists", "discovery", "arxiv", "nature",
            "екзопланет", "комет", "астероїд", "космос", "дослідженн",
            "комета", "астероид", "космос", "исследован",
        ]

        # Tavily's `topic` enum is not stable across API versions; keep values conservative.
        if any(k in combined for k in finance_keywords):
            return "general"
        if any(k in combined for k in news_keywords):
            return "news"
        if any(k in combined for k in science_keywords):
            # Science claims are often about recent discoveries; bias towards news.
            return "news"
        return "general"

    def _detect_time_filters(self, *, ttl: int | None, query: str) -> dict:
        # If cache TTL is very short, this is likely a time-sensitive check.
        # Use a narrow window to avoid irrelevant/old SEO content.
        is_time_sensitive = bool(ttl is not None and ttl <= 1800)
        if not is_time_sensitive:
            return {}

        # Tavily supports `time_range` but not all deployments accept extra fields like `days`.
        # Keep payload conservative to avoid 400 Bad Request.
        return {"time_range": "week"}

    def _get_exclude_domains(self) -> list[str]:
        raw = os.getenv("SPECTRUE_TAVILY_EXCLUDE_DOMAINS", "").strip()
        if not raw:
            return []
        parts = [p.strip().lower() for p in re.split(r"[,\n]", raw) if p.strip()]
        # Bound to avoid oversized payloads.
        out: list[str] = []
        for d in parts:
            d = d.lstrip(".")
            if d and d not in out:
                out.append(d)
            if len(out) >= 150:
                break
        return out

    def _raw_content_mode(self, *, depth: str, domains: list[str] | None) -> bool:
        """
        Tavily param `include_raw_content`: bool.
        Default is cost-aware: enabled only for non-domain-filtered advanced searches (deep dive).
        """
        val = os.getenv("SPECTRUE_TAVILY_INCLUDE_RAW_CONTENT", "auto").strip().lower()
        if val in ("0", "false", "no", "off"):
            return False
        if val in ("1", "true", "yes", "on"):
            return True

        # auto
        if depth == "advanced" and not domains:
            return True
        return False

    def _raw_max_results(self) -> int:
        try:
            n = int(os.getenv("SPECTRUE_TAVILY_RAW_MAX_RESULTS", "4"))
        except Exception:
            n = 4
        return max(1, min(n, 10))

    def _quality_from_ranked(self, ranked: list[dict]) -> dict:
        rels = [
            float(r.get("relevance_score", 0.0) or 0.0)
            for r in (ranked or [])
            if isinstance(r.get("relevance_score"), (int, float))
        ]
        top = rels[:5]
        avg = sum(top) / len(top) if top else 0.0
        best = max(rels) if rels else 0.0
        n = len(ranked or [])
        poor = (n < 3) or (avg < 0.22) or (best < 0.28)
        return {
            "sources_count": n,
            "avg_relevance_top5": avg,
            "best_relevance": best,
            "quality": "poor" if poor else "good",
        }

    def _should_fulltext_enrich(self, *, quality: str, depth: str) -> bool:
        val = os.getenv("SPECTRUE_FULLTEXT_FETCH", "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
        if val in ("0", "false", "no", "off"):
            return False
        # Default OFF to avoid server-side crawling / IP reputation issues.
        return False

    async def _fetch_extract_text(self, url: str) -> str | None:
        if not self._is_valid_url(url):
            return None

        cache_key = f"page|{url}"
        try:
            if cache_key in self.page_cache:
                cached = self.page_cache[cache_key]
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception:
            pass

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SpectrueBot/1.0; +https://spectrue.net)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            resp = await self.client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None

            raw = resp.content
            if raw and len(raw) > 2_000_000:
                raw = raw[:2_000_000]

            html = raw.decode(resp.encoding or "utf-8", errors="ignore")
            if not html.strip():
                return None

            # Import lazily (keeps engine lightweight if not needed)
            import trafilatura

            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if not extracted:
                return None

            cleaned = re.sub(r"\s+", " ", extracted).strip()
            if len(cleaned) < 200:
                return None

            cleaned = cleaned[:3000]
            try:
                self.page_cache.set(cache_key, cleaned, expire=self.page_ttl)
            except Exception:
                pass
            return cleaned
        except Exception:
            return None

    async def _enrich_with_fulltext(self, query: str, ranked: list[dict], *, limit: int = 3) -> tuple[list[dict], int]:
        urls = [r.get("url") for r in (ranked or []) if r.get("url")]
        urls = urls[: max(0, int(limit))]
        if not urls:
            return ranked, 0

        fetched = 0
        tasks = [self._fetch_extract_text(u) for u in urls]
        texts = await asyncio.gather(*tasks)

        updated: list[dict] = []
        for item in ranked:
            url = item.get("url")
            if url in urls:
                idx = urls.index(url)
                full = texts[idx]
                if full:
                    new_item = dict(item)
                    new_item["content"] = full
                    # Recompute relevance with richer content.
                    new_item["relevance_score"] = self._relevance_score(
                        query,
                        new_item.get("title", ""),
                        full,
                        url,
                        tavily_score=new_item.get("score"),
                    )
                    new_item["fulltext"] = True
                    updated.append(new_item)
                    fetched += 1
                    continue
            updated.append(item)

        updated = self._rank_and_filter(query, updated)
        return updated, fetched

    async def _request_search(
        self,
        q: str,
        depth: str,
        limit: int,
        *,
        ttl: int | None,
        domains: list[str] | None = None,
        country: str | None = None,
    ) -> dict:
        raw_mode = self._raw_content_mode(depth=depth, domains=domains)
        effective_limit = min(limit, self._raw_max_results()) if raw_mode else limit

        payload = {"query": q, "search_depth": depth, "max_results": effective_limit}
        # M20: Add trusted domains filter for Tier 1 searches
        if domains:
            payload["include_domains"] = domains

        exclude_domains = self._get_exclude_domains()
        if exclude_domains:
            # Do not exclude domains that are explicitly included (Tier 1).
            include_set = {d.lower().lstrip(".") for d in (domains or [])}
            filtered = [d for d in exclude_domains if d not in include_set]
            if filtered:
                payload["exclude_domains"] = filtered

        payload["topic"] = self._detect_topic(q)
        payload.update(self._detect_time_filters(ttl=ttl, query=q))
        if raw_mode:
            payload["include_raw_content"] = True

        if country and isinstance(country, str):
            c = country.strip()
            if c and not re.fullmatch(r"[A-Za-z]{2}", c) and len(c) <= 64:
                payload["country"] = c
        
        url = "https://api.tavily.com/search"
        async with self._sem:
            Trace.event("tavily.request", {"url": url, "payload": payload})
            r = await self.client.post(url, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Tavily can respond with 400 if optional fields drift from the API schema.
                # Retry once with a minimal payload before giving up.
                if e.response is not None and e.response.status_code == 400:
                    def _safe_payload_for_log(p: dict) -> dict:
                        out = dict(p or {})
                        qv = out.get("query")
                        if isinstance(qv, str) and len(qv) > 220:
                            out["query"] = qv[:220] + "…"
                        for k in ("include_domains", "exclude_domains"):
                            v = out.get(k)
                            if isinstance(v, list) and len(v) > 30:
                                out[k] = v[:30] + [f"...(+{len(v) - 30} more)"]
                        return out

                    try:
                        logger.error("[Tavily] 400 Error Payload: %s", _safe_payload_for_log(payload))
                        logger.error("[Tavily] 400 Response: %s", (e.response.text or "")[:800])
                    except Exception:
                        # Never fail the request flow due to logging.
                        pass

                    logger.debug(
                        "[Tavily] 400 Bad Request. Retrying with minimal payload. Response: %s",
                        (e.response.text or "")[:500],
                    )
                    minimal = {
                        "query": q,
                        "search_depth": depth,
                        "max_results": effective_limit,
                    }
                    if domains:
                        minimal["include_domains"] = domains
                    if exclude_domains:
                        include_set = {d.lower().lstrip(".") for d in (domains or [])}
                        filtered = [d for d in exclude_domains if d not in include_set]
                        if filtered:
                            minimal["exclude_domains"] = filtered
                    Trace.event("tavily.request.retry_minimal", {"url": url, "payload": minimal})
                    r2 = await self.client.post(url, json=minimal)
                    try:
                        r2.raise_for_status()
                    except httpx.HTTPStatusError as e2:
                        if e2.response is not None and e2.response.status_code == 400:
                            try:
                                logger.error("[Tavily] 400 Error Payload (minimal): %s", _safe_payload_for_log(minimal))
                                logger.error("[Tavily] 400 Response (minimal): %s", (e2.response.text or "")[:800])
                            except Exception:
                                pass
                        raise
                    Trace.event(
                        "tavily.response",
                        {"status_code": r2.status_code, "text": r2.text},
                    )
                    return r2.json()

                # Include response text for easier debugging (do not include payload to avoid leaking config).
                if e.response is not None:
                    logger.debug(
                        "[Tavily] HTTP error %s. Response: %s",
                        e.response.status_code,
                        (e.response.text or "")[:500],
                    )
                raise
            Trace.event("tavily.response", {"status_code": r.status_code, "text": r.text})
            return r.json()

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        ttl: int | None = None,
        domains: list[str] | None = None,
        lang: str | None = None,
        country: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Search with Tavily and return both context text and structured sources.
        
        Args:
            query: Search query string
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results (default 7 for M20)
            ttl: Cache TTL in seconds
            domains: Optional list of domains to restrict search to (M20: for Tier 1 searches)
        
        Returns:
            tuple: (context_str, sources_list) where sources_list contains dicts with title/url/content
        """
        if not self.api_key:
            # Avoid slow network round-trips just to fail auth.
            logger.debug("[Tavily] API key missing; skipping search")
            return ("", [])

        if not query or not isinstance(query, str) or len(query.strip()) < 3:
            logger.debug("[Tavily] Query too short or empty, skipping search")
            return ("", [])

        q = " ".join(query.split()).strip()
        depth, limit = self._normalize_params(search_depth, max_results)
        raw_mode = self._raw_content_mode(depth=depth, domains=domains)
        effective_limit = min(limit, self._raw_max_results()) if raw_mode else limit
        country = country
        Trace.event(
            "tavily.search.start",
            {
                "query": q,
                "depth": depth,
                "limit": effective_limit,
                "raw_content_mode": bool(raw_mode),
                "domains_count": len(domains or []),
                "country": country,
            },
        )
        if country:
            if is_local_run():
                logger.info("[Tavily] Using country filter: %s", country)
            else:
                logger.debug("[Tavily] Using country filter: %s", country)
        domains_key = ""
        if domains:
            # Cache key must reflect domain filtering; keep it stable & bounded.
            domains_key = ",".join(sorted(set(domains)))[:2000]
        cache_key = f"{q}|{depth}|{effective_limit}|{int(bool(raw_mode))}|{country or ''}|{domains_key}"

        effective_ttl = ttl if ttl is not None else self.ttl
        self.last_cache_hit = False

        if cache_key in self.cache:
            self.last_cache_hit = True
            cached_data = self.cache[cache_key]
            # Cache stores tuple (context_str, sources_list)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                context_str, sources_list = cached_data
                Trace.event(
                    "tavily.cache.hit",
                    {
                        "query": q,
                        "depth": depth,
                        "limit": effective_limit,
                        "raw_content_mode": bool(raw_mode),
                        "domains_count": len(domains or []),
                        "country": country,
                        "context_chars": len(context_str or ""),
                        "sources_count": len(sources_list or []),
                    },
                )
                Trace.event(
                    "tavily.search.done",
                    {
                        "query": q,
                        "depth": depth,
                        "limit": effective_limit,
                        "raw_content_mode": bool(raw_mode),
                        "domains_count": len(domains or []),
                        "country": country,
                        "cache_hit": True,
                        "context_chars": len(context_str or ""),
                        "sources_count": len(sources_list or []),
                    },
                )
                logger.debug(
                    "[Tavily] ✓ Cache hit for '%s...' (%d chars, %d sources)",
                    q[:100],
                    len(context_str),
                    len(sources_list),
                )
                return (context_str, sources_list)
            else:
                # Old cache format (string only), invalidate
                logger.debug("[Tavily] Cache format outdated, re-fetching")
                del self.cache[cache_key]
        Trace.event(
            "tavily.cache.miss",
            {
                "query": q,
                "depth": depth,
                "limit": effective_limit,
                "raw_content_mode": bool(raw_mode),
                "domains_count": len(domains or []),
                "country": country,
            },
        )

        try:
            domain_info = f", domains={len(domains)}" if domains else ""
            logger.debug("[Tavily] Searching: '%s...' (depth=%s, limit=%s%s)", q[:100], depth, limit, domain_info)
            response = await self._request_search(
                q,
                depth,
                effective_limit,
                ttl=ttl,
                domains=domains,
                country=country,
            )
            results_raw = response.get('results', [])
            logger.debug("[Tavily] Got %d raw results", len(results_raw))
            
            cleaned = self._clean_results(results_raw)
            ranked = self._rank_and_filter(q, cleaned)
            logger.debug("[Tavily] After cleaning: %d results", len(cleaned))
            logger.debug("[Tavily] After ranking/filter: %d results", len(ranked))

            quality = self._quality_from_ranked(ranked)
            fulltext_fetches = 0
            if self._should_fulltext_enrich(quality=quality["quality"], depth=depth):
                ranked, fulltext_fetches = await self._enrich_with_fulltext(q, ranked, limit=3)
                quality = self._quality_from_ranked(ranked)
                logger.debug("[Tavily] Fulltext enrichment: %d pages", fulltext_fetches)

            self.last_search_meta = {
                "provider": "tavily",
                "query": q,
                "depth": depth,
                "domains_filtered": bool(domains),
                "cache_hit": bool(self.last_cache_hit),
                "page_fetches": fulltext_fetches,
                "topic": self._detect_topic(q),
                **self._detect_time_filters(ttl=ttl, query=q),
                "raw_content_mode": raw_mode,
                "raw_max_results": self._raw_max_results() if raw_mode else None,
                "country": country,
                **quality,
            }
            
            # M29: Check if URL is from a trusted domain
            def is_trusted_url(url: str) -> bool:
                try:
                    host = urlparse(url).netloc.lower()
                    if host.startswith("www."):
                        host = host[4:]
                    # Check direct match or parent domain
                    if host in TRUSTED_DOMAINS:
                        return True
                    parts = host.split(".")
                    for i in range(len(parts) - 1):
                        if ".".join(parts[i:]) in TRUSTED_DOMAINS:
                            return True
                except Exception:
                    pass
                return False
            
            # Generate context text for LLM with trust markers
            def format_source(obj: dict) -> str:
                trust_tag = "[TRUSTED] " if is_trusted_url(obj['url']) else ""
                rel = obj.get("relevance_score")
                rel_tag = f"[REL={rel:.2f}] " if isinstance(rel, (int, float)) else ""
                raw_tag = "[RAW] " if obj.get("raw_content_used") else ""
                return f"Source: {trust_tag}{raw_tag}{rel_tag}{obj['title']}\nURL: {obj['url']}\nContent: {obj['content']}\n---"
            
            context = "\n".join([format_source(obj) for obj in ranked])
            
            # Create structured sources for frontend
            sources_list = [
                {
                    "title": obj["title"],
                    "link": obj["url"],
                    "snippet": obj["content"],
                    "relevance_score": obj.get("relevance_score"),
                    "origin": "WEB"
                }
                for obj in ranked
            ]
            
            logger.debug("[Tavily] Generated context: %d chars, %d sources", len(context), len(sources_list))
            if ranked and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Tavily] Sample result: %s...", ranked[0]['title'][:80])
            
            # Cache both context and sources
            if context:
                self.cache.set(cache_key, (context, sources_list), expire=effective_ttl)

            Trace.event(
                "tavily.search.done",
                {
                    "query": q,
                    "depth": depth,
                    "limit": effective_limit,
                    "raw_content_mode": bool(raw_mode),
                    "domains_count": len(domains or []),
                    "country": country,
                    "cache_hit": bool(self.last_cache_hit),
                    "context_chars": len(context),
                    "sources_count": len(sources_list),
                    "quality": quality.get("quality"),
                    "best_relevance": quality.get("best_relevance"),
                    "avg_relevance_top5": quality.get("avg_relevance_top5"),
                    "page_fetches": fulltext_fetches,
                },
            )
            
            return (context, sources_list)
        except Exception as e:
            logger.warning("[Tavily] ✗ Error during search (%s): %s", depth, e)
            Trace.event(
                "tavily.search.error",
                {
                    "query": q if "q" in locals() else "",
                    "depth": depth,
                    "limit": effective_limit if "effective_limit" in locals() else None,
                    "raw_content_mode": bool(raw_mode) if "raw_mode" in locals() else None,
                    "domains_count": len(domains or []),
                    "country": country,
                    "error": str(e),
                },
            )
            self.last_search_meta = {
                "provider": "tavily",
                "query": q if "q" in locals() else "",
                "depth": search_depth,
                "domains_filtered": bool(domains),
                "cache_hit": False,
                "page_fetches": 0,
                "sources_count": 0,
                "avg_relevance_top5": 0.0,
                "best_relevance": 0.0,
                "quality": "poor",
            }
            return ("", [])
