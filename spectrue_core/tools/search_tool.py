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

import asyncio
import httpx
import logging
import re
from pathlib import Path
import diskcache
from typing import Tuple
from urllib.parse import urlparse
from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.trace import Trace

from spectrue_core.verification.trusted_sources import ALL_TRUSTED_DOMAINS as TRUSTED_DOMAINS

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)


class WebSearchTool:
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        self.api_key = config.tavily_api_key if config else None

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

        conc = int(getattr((config.runtime.search if config else None), "tavily_concurrency", 8) or 8)
        self._sem = asyncio.Semaphore(max(1, min(conc, 16)))
        self.last_cache_hit: bool = False
        self.last_search_meta: dict = {}

    def _normalize_params(self, search_depth: str, max_results: int) -> Tuple[str, int]:
        depth = (search_depth or "basic").lower()
        if depth not in ("basic", "advanced"):
            depth = "basic"
        m = int(max_results) if max_results else 5  # M46: Reduced to 5 for better relevance
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

    def _is_trusted_host(self, host: str) -> bool:
        """Centralized trusted host check with subdomain walking."""
        if not host: 
            return False
        h = host.lower()
        if h.startswith("www."):
            h = h[4:]
        
        # Exact match
        if h in TRUSTED_DOMAINS:
            return True
        
        # Safety net: Do not trust known public suffixes during parent walk
        # to prevent accidental whitelisting of "co.uk" or similar.
        forbidden_suffixes = {"co.uk", "com.au", "co.jp", "com.br", "co.id", "co.in", "com.cn", "org.uk"}

        # Parent walk
        parts = h.split(".")
        for i in range(len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in forbidden_suffixes:
                continue
            if candidate in TRUSTED_DOMAINS:
                return True
        return False

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
        
        # Penalize extremely short snippets (often random/empty)
        length_penalty = 0.0
        if content and len(content) < 80:
            length_penalty = 0.08

        # Small bonus for trusted domains (stability)
        trusted_bonus = 0.0
        try:
            host = urlparse(url).netloc
            if self._is_trusted_host(host):
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
                         final_score = max(lexical_score, ts * 0.70) # Reduced trust in pure semantic match
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
                final_score *= 0.5 # Heavy penalty
        
        # M36: Hard penalize specific forum patterns in known domains
        low_url = (url or "").lower()
        if "forum.nasaspaceflight.com" in low_url:
             final_score = 0.0

        return max(0.0, min(1.0, final_score))

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
        # M46: Increased threshold slightly
        kept = [r for r in scored if r.get("relevance_score", 0.0) >= 0.15]
        if not kept:
            # Fallback for borderline items, but never return pure junk (0.0)
            kept = [r for r in scored if r.get("relevance_score", 0.0) >= 0.05]
        
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

    def _detect_time_filters(self, *, ttl: int | None, query: str) -> dict:
        # Disabled: time_range="week" kills science/historical cases.
        # Tavily returns fresher results by default anyway.
        return {}

    def _get_exclude_domains(self) -> list[str]:
        try:
            out = list((self.config.runtime.search.tavily_exclude_domains or []) if self.config else [])
        except Exception:
            out = []
        return out

    def _raw_content_mode(self, *, depth: str, domains: list[str] | None) -> bool:
        """
        Tavily param `include_raw_content`: bool.
        Default is cost-aware: enabled only for non-domain-filtered advanced searches (deep dive).
        """
        try:
            forced = self.config.runtime.search.tavily_include_raw_content if self.config else None
        except Exception:
            forced = None
        if forced is True:
            return True
        if forced is False:
            return False

        # auto:
        if depth == "advanced" and not domains:
            return True
        return False

    def _raw_max_results(self) -> int:
        try:
            n = int(self.config.runtime.search.tavily_raw_max_results if self.config else 4)
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
        try:
            return bool(self.config.runtime.features.fulltext_fetch if self.config else False)
        except Exception:
            return False

    def _clean_url_key(self, url: str) -> str:
        """Strip tracking params/fragments for cache key reuse."""
        try:
            u = urlparse(url)
            # Filter query params
            q = []
            if u.query:
                # remove common tracking garbage
                ignored = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}
                pairs = u.query.split('&')
                valid_pairs = [p for p in pairs if p.split('=')[0].lower() not in ignored]
                q = "&".join(sorted(valid_pairs))
            
            # Reconstruct without fragment
            # scheme://netloc/path;params?query
            qs = f"?{q}" if q else ""
            return f"{u.scheme}://{u.netloc}{u.path}{qs}"
        except Exception:
            return url

    async def _fetch_extract_text(self, url: str) -> str | None:
        """
        Securely fetch/extract text from a URL using Tavily Extract API.
        No local HTTP requests to the target URL are made from this server.
        """
        if not self._is_valid_url(url) or not self.api_key:
            return None

        # Clean URL for cache key to improve hit rate and reduce duplication
        clean_url = self._clean_url_key(url)
        cache_key = f"page_tavily|{clean_url}"
        try:
            if cache_key in self.page_cache:
                cached = self.page_cache[cache_key]
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception:
            pass
        
        # Tavily Extract API
        endpoint = "https://api.tavily.com/extract"
        payload = {
            "urls": [url],
            # Use text format - smaller input (5.6k vs 13.9k), LLM cleans equally well
            "format": "text",
        }
        
        try:
            async with self._sem:
                Trace.event("tavily.extract.request", {"url": endpoint, "target": url})
                r = await self.client.post(endpoint, json=payload)
                r.raise_for_status()
                data = r.json()
                
            # Response format: {"results": [{"url": "...", "raw_content": "..."}]}
            results = data.get("results", [])
            if not results:
                return None
                
            # Find matching result
            extracted_text = ""
            for item in results:
                if item.get("url") == url:
                    extracted_text = item.get("raw_content") or item.get("content") or ""
                    break
            
            # Fallback to first if no explicit URL match (Tavily sometimes normalizes input URL)
            if not extracted_text and results:
                extracted_text = results[0].get("raw_content") or results[0].get("content") or ""
            
            if not extracted_text:
                return None

            cleaned = ""
            # M47: Use trafilatura for robust HTML extraction (removes navbar/footer/ads)
            if trafilatura and ("<html" in extracted_text or "<body" in extracted_text or "<div" in extracted_text):
                try:
                    # include_links=True to keep sources for verification
                    cleaned = trafilatura.extract(extracted_text, include_links=True, include_images=False, include_comments=False)
                except Exception as e:
                    logger.warning("[Tavily] Trafilatura extraction failed: %s", e)
            
            if not cleaned:
                # Fallback to simple cleaning if trafilatura failed or not HTML
                cleaned = extracted_text
            
            # Normalize: just strip, DON'T flatten whitespace (re.sub \s+ " ")
            # We want to keep paragraphs/newlines if trafilatura preserved them.
            cleaned = cleaned.strip()
            
            if len(cleaned) < 50: # Minimum useful content
                return None

            # Cache the result
            try:
                self.page_cache.set(cache_key, cleaned, expire=self.page_ttl)
            except Exception:
                pass
            
            Trace.event("tavily.extract.success", {"url": url, "chars": len(cleaned)})
            return cleaned

        except Exception as e:
            logger.warning("[Tavily] Extract failed for %s: %s", url, e)
            Trace.event("tavily.extract.error", {"url": url, "error": str(e)})
            return None


    async def _enrich_with_fulltext(self, query: str, ranked: list[dict], *, limit: int = 3) -> tuple[list[dict], int]:
        target_urls = [r.get("url") for r in (ranked or []) if r.get("url")]
        target_urls = target_urls[: max(0, int(limit))]
        if not target_urls:
            return ranked, 0

        fetched = 0
        tasks = [self._fetch_extract_text(u) for u in target_urls]
        texts = await asyncio.gather(*tasks)
        
        # Map URL -> extracted text (O(1) lookup)
        url_map = {u: t for u, t in zip(target_urls, texts) if t}

        updated: list[dict] = []
        for item in ranked:
            url = item.get("url")
            if url in url_map:
                full = url_map[url]
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
            else:
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
        exclude_domains: list[str] | None = None,
    ) -> dict:
        raw_mode = self._raw_content_mode(depth=depth, domains=domains)
        effective_limit = min(limit, self._raw_max_results()) if raw_mode else limit

        payload = {"query": q, "search_depth": depth, "max_results": effective_limit}
        # M20: Add trusted domains filter for Tier 1 searches
        if domains:
            payload["include_domains"] = domains

        # Merge config exclusions with method-level exclusions
        global_exclude = self._get_exclude_domains()
        merged_exclude = set(global_exclude)
        if exclude_domains:
            merged_exclude.update(exclude_domains)
        
        # Determine final exclusion list
        final_exclude = list(merged_exclude)
        
        if final_exclude:
            # Do not exclude domains that are explicitly included (Tier 1).
            include_set = {d.lower().lstrip(".") for d in (domains or [])}
            filtered = [d for d in final_exclude if d not in include_set]
            if filtered:
                payload["exclude_domains"] = filtered

        payload["topic"] = "general"
        payload.update(self._detect_time_filters(ttl=ttl, query=q))
        if raw_mode:
            payload["include_raw_content"] = True

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
        max_results: int = 5,
        ttl: int | None = None,
        domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        lang: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Search with Tavily and return both context text and structured sources.
        
        Args:
            query: Search query string
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results (default 7 for M20)
            ttl: Cache TTL in seconds
            ttl: Cache TTL in seconds
            domains: Optional list of domains to restrict search to (M20: for Tier 1 searches)
            exclude_domains: Optional list of domains to exclude (e.g. for Tier 2)
        
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
        Trace.event(
            "tavily.search.start",
            {
                "query": q,
                "depth": depth,
                "limit": effective_limit,
                "raw_content_mode": bool(raw_mode),
                "domains_count": len(domains or []),
            },
        )
        domains_key = ""
        if domains:
            # Cache key must reflect domain filtering; keep it stable & bounded.
            domains_key = ",".join(sorted(set(domains)))[:2000]
        
        exclude_key = ""
        if exclude_domains:
            # Hash of all unique exclude domains for cache key (M48)
            # M53: Ensure no None in exclude_domains list (can come from call-site)
            clean_exclude_domains = [d for d in exclude_domains if d is not None]
            exclude_key = ",".join(sorted(set(clean_exclude_domains)))[:2000]
        cache_key = f"{q}|{depth}|{effective_limit}|{int(bool(raw_mode))}|{domains_key}|{exclude_key}"

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
                exclude_domains=exclude_domains,
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
                "topic": "general",
                **self._detect_time_filters(ttl=ttl, query=q),
                "raw_content_mode": raw_mode,
                "raw_max_results": self._raw_max_results() if raw_mode else None,
                **quality,
            }
            
            # Generate context text for LLM WITHOUT trusted/rel/raw tags in the content
            def format_source(obj: dict) -> str:
                return f"Source: {obj['title']}\nURL: {obj['url']}\nContent: {obj['content']}\n---"
            
            context = "\n".join([format_source(obj) for obj in ranked])
            
            # Create structured sources for frontend
            def safe_is_trusted(u: str) -> bool:
                try:
                    return self._is_trusted_host(urlparse(u).netloc)
                except Exception:
                    return False

            sources_list = [
                {
                    "title": obj["title"],
                    "link": obj["url"],
                    "snippet": obj["content"],
                    "relevance_score": obj.get("relevance_score"),
                    "is_trusted": safe_is_trusted(obj.get("url", "")),
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

    async def close(self):
        """Cleanup resources."""
        if self.client:
            await self.client.aclose()
        if self.cache:
            self.cache.close()
        if self.page_cache:
            self.page_cache.close()
