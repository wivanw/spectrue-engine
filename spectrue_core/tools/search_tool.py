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
        seen: set[str] = set()
        for obj in (results or []):
            title = (obj.get("title") or "").strip()
            url = (obj.get("url") or "").strip()
            content = (obj.get("content") or "").strip()
            raw_content = (obj.get("raw_content") or "").strip()
            score = obj.get("score")  # Tavily may provide relevance score

            if not self._is_valid_url(url):
                logger.debug("[Search] Discarded invalid URL: %s", url)
                continue

            # Fallback title to domain to avoid "Без заголовка"
            if not title:
                try:
                    title = urlparse(url).netloc or url
                except Exception:
                    title = url

            # Improved deduplication by canonical URL (Step 287 fix)
            # Remove protocol, www, query params, and fragment
            u = urlparse(url)
            clean_url = re.sub(r'^(www\.)?', '', (u.netloc or "").lower()) + (u.path or "").rstrip('/')
            
            if clean_url in seen:
                # If we've seen this URL, keep the one with better title/content?
                # For simplicity, keep first encountered (usually higher rank from Tavily)
                logger.debug("[Search] Discarded URL duplicate: %s", url)
                continue
            seen.add(clean_url)

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
        # Keep best results; drop clearly off-topic tail.
        # M46: Increased threshold slightly
        # Keep best results; drop clearly off-topic tail.
        # M60: Advanced Noise Filtering to remove "poisonous" archives (UN, legal PDFs)
        kept = []
        discarded_reasons = []

        # Domains that often return historic/legal noise for "blockade" queries
        ARCHIVE_DOMAINS = {"legal.un.org", "docs.un.org", "press.un.org", "undocs.org", "history.state.gov", "oecd.org"}
        # Keywords that MUST be present to redeem an archive/PDF document
        MUST_HAVE_KEYWORDS = {"trump", "truth social", "biden", "zelensky", "putin", "breaking", "news", "2024", "2025"}

        for r in scored:
            score = r.get("relevance_score", 0.0)
            url = r.get("url", "").lower()
            domain = urlparse(url).netloc
            content_lower = (r.get("content") or "").lower()
            
            # Check for generic PDF/Archive junk
            is_noise_domain = domain in ARCHIVE_DOMAINS or "legal.un.org" in url
            is_pdf = ".pdf" in url

            if is_noise_domain or is_pdf:
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

    # M58: _detect_time_filters removed - was disabled and always returned {}

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

    def _should_fulltext_enrich(self) -> bool:
        """Check if fulltext enrichment is enabled in config."""
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
            # M60: Use markdown format to preserve inline URLs for source extraction
            "format": "markdown",
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
        max_results: int,
        *,
        domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        raw_mode: bool = False,
    ) -> dict:
        payload = {"query": q, "search_depth": depth, "max_results": max_results}
        # M20: Add trusted domains filter for Tier 1 searches
        if domains:
            payload["include_domains"] = domains

        # Merge config exclusions with method-level exclusions
        global_exclude = self._get_exclude_domains()
        merged_exclude = set(global_exclude)
        if exclude_domains:
            merged_exclude.update(exclude_domains)
        
        # Determine final exclusion list (sorted for deterministic behavior)
        final_exclude = sorted({d.lower().lstrip(".") for d in merged_exclude if d})
        
        if final_exclude:
            # Do not exclude domains that are explicitly included (Tier 1).
            include_set = {d.lower().lstrip(".") for d in (domains or [])}
            filtered = [d for d in final_exclude if d not in include_set]
            # Cap to 32 domains to prevent payload size issues with Tavily API
            if len(filtered) > 32:
                filtered = filtered[:32]
            if filtered:
                payload["exclude_domains"] = filtered

        payload["topic"] = "general"
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
                        "max_results": max_results,
                    }
                    if domains:
                        minimal["include_domains"] = domains
                    
                    # NOTE: For minimal retry we intentionally DO NOT include method-level exclude_domains
                    # (these often cause schema/size issues and lead to the same 400 again).
                    # But we DO keep config-level global excludes to preserve project policy.
                    try:
                        retry_exclude = list(self._get_exclude_domains() or [])
                    except Exception:
                        retry_exclude = []
                    
                    if retry_exclude:
                        include_set = {d.lower().lstrip(".") for d in (domains or [])}
                        ex = sorted({d.lower().lstrip(".") for d in retry_exclude if d})
                        ex = [d for d in ex if d not in include_set]
                        if ex:
                            minimal["exclude_domains"] = ex[:32]
                    
                    logger.debug("[Tavily] Minimal retry exclude_domains=%d", len(minimal.get("exclude_domains") or []))
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
        num_results: int = 5,
        depth: str = "basic",
        raw_content: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """Search with Tavily and return both context text and structured sources.
        
        Args:
            query: Search query string
            num_results: Maximum number of results (default 5)
            depth: "basic" or "advanced"
            raw_content: Force raw content mode
            include_domains: Optional list of domains to restrict search to (Tier 1 searches)
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

        # Normalize params once here, pass already-normalized to internal
        _d, _l = self._normalize_params(depth, num_results)
        
        # Delegate to internal implementation
        return await self._search_internal(
            query=query,
            num_results=_l,
            depth=_d,
            raw_content=raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains
        )

    async def _search_internal(
        self,
        query: str,
        num_results: int = 5,
        depth: str = "basic",
        raw_content: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        ttl: int | None = None, 
    ) -> tuple[str, list[dict]]: 
        from spectrue_core.utils.text_processing import normalize_search_query
        q = normalize_search_query(query)
        
        # Params already normalized by caller, use directly
        limit = num_results
        
        # Normalize include_domains FIRST (before raw_mode calculation)
        normalized_include: list[str] | None = None
        if include_domains:
            seen_inc: set[str] = set()
            temp_inc: list[str] = []
            for d in include_domains:
                if not d:
                    continue
                dd = d.lower().lstrip(".")
                if dd not in seen_inc:
                    seen_inc.add(dd)
                    temp_inc.append(dd)
            normalized_include = temp_inc
        
        # raw_content param can force raw mode, otherwise use heuristic (with normalized domains)
        raw_mode = bool(raw_content) or self._raw_content_mode(depth=depth, domains=normalized_include)
        effective_limit = min(limit, self._raw_max_results()) if raw_mode else limit
        
        Trace.event(
            "tavily.search.start",
            {
                "query": q,
                "depth": depth,
                "limit": effective_limit,
                "raw_content_mode": bool(raw_mode),
                "domains_count": len(normalized_include or []),
            },
        )
        
        domains_key = ""
        if normalized_include:
            # Cache key must reflect domain filtering; keep it stable & bounded.
            domains_key = ",".join(sorted(normalized_include))[:2000]
        
        # Normalize and cap exclude_domains to match what actually goes to API
        normalized_exclude: list[str] | None = None
        if exclude_domains:
            seen_ex: set[str] = set()
            temp_ex: list[str] = []
            for d in exclude_domains:
                if not d:
                    continue
                dd = d.lower().lstrip(".")
                if dd not in seen_ex:
                    seen_ex.add(dd)
                    temp_ex.append(dd)
            # Cap to 32 and sort for deterministic behavior
            temp_ex_sorted = sorted(temp_ex)
            normalized_exclude = temp_ex_sorted[:32] if len(temp_ex_sorted) > 32 else temp_ex_sorted
        
        exclude_key = ""
        if normalized_exclude:
            exclude_key = ",".join(normalized_exclude)  # already sorted
        
        # M58: Normalize cache key with lowercase for better hit rate
        # "Kyiv population" and "kyiv Population" should hit the same cache entry
        q_cache = q.lower()
        cache_key = f"{q_cache}|{depth}|{effective_limit}|{int(bool(raw_mode))}|{domains_key}|{exclude_key}"

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
                        "domains_count": len(normalized_include or []),
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
                        "domains_count": len(normalized_include or []),
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
                "domains_count": len(normalized_include or []),
            },
        )

        try:
            domain_info = f", domains={len(normalized_include or [])}" if normalized_include else ""
            logger.debug("[Tavily] Searching: '%s...' (depth=%s, limit=%s%s)", q[:100], depth, effective_limit, domain_info)
            response = await self._request_search(
                q,
                depth,
                effective_limit,
                domains=normalized_include,
                exclude_domains=normalized_exclude,
                raw_mode=raw_mode,
            )
            results_raw = response.get('results', [])
            logger.debug("[Tavily] Got %d raw results", len(results_raw))
            
            cleaned = self._clean_results(results_raw)
            ranked = self._rank_and_filter(q, cleaned)
            logger.debug("[Tavily] After cleaning: %d results", len(cleaned))
            logger.debug("[Tavily] After ranking/filter: %d results", len(ranked))

            # M61: Second-pass diversification if Tavily returns too many same-domain results.
            # If we got <3 unique domains, try excluding the dominant domain once.
            def _domain(u: str) -> str:
                try:
                    d = (urlparse(u).netloc or "").lower()
                    return d[4:] if d.startswith("www.") else d
                except Exception:
                    return ""

            uniq = {_domain(r.get("url", "")) for r in (ranked or []) if r.get("url")}
            uniq.discard("")
            if len(uniq) < 3 and ranked:
                # Dominant domain = domain of the top-ranked item
                dom0 = _domain(ranked[0].get("url", ""))
                if dom0:
                    logger.debug("[Tavily] Diversify pass: excluding dominant domain=%s", dom0)
                    try:
                        response2 = await self._request_search(
                            q,
                            depth,
                            effective_limit,
                            domains=normalized_include,
                            exclude_domains=(normalized_exclude or []) + [dom0],
                            raw_mode=raw_mode,
                        )
                        results2 = self._clean_results(response2.get("results", []))
                        merged = cleaned + results2
                        ranked = self._rank_and_filter(q, merged)
                        logger.debug("[Tavily] Diversify pass done: %d results", len(ranked))
                    except Exception as div_err:
                        logger.debug("[Tavily] Diversify pass failed: %s", div_err)

            quality = self._quality_from_ranked(ranked)
            fulltext_fetches = 0
            if self._should_fulltext_enrich():
                ranked, fulltext_fetches = await self._enrich_with_fulltext(q, ranked, limit=3)
                quality = self._quality_from_ranked(ranked)
                logger.debug("[Tavily] Fulltext enrichment: %d pages", fulltext_fetches)

            self.last_search_meta = {
                "provider": "tavily",
                "query": q,
                "depth": depth,
                "domains_filtered": bool(normalized_include),
                "cache_hit": bool(self.last_cache_hit),
                "page_fetches": fulltext_fetches,
                "topic": "general",
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
                    "domains_count": len(normalized_include or []),
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
            # Log with exception type for better debugging (str(e) can be empty)
            err_type = type(e).__name__
            err_msg = str(e) or repr(e)
            logger.warning("[Tavily] ✗ Error %s during search (%s): %r", err_type, depth, e)
            Trace.event(
                "tavily.search.error",
                {
                    "query": q if "q" in locals() else "",
                    "depth": depth,
                    "limit": effective_limit if "effective_limit" in locals() else None,
                    "raw_content_mode": bool(raw_mode) if "raw_mode" in locals() else None,
                    "domains_count": len(normalized_include or []),
                    "error_type": err_type,
                    "error": err_msg,
                },
            )
            self.last_search_meta = {
                "provider": "tavily",
                "query": q if "q" in locals() else "",
                "depth": depth,
                "domains_filtered": bool(normalized_include),
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
