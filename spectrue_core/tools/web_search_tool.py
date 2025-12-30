import asyncio
import logging
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import diskcache

from spectrue_core.config import SpectrueConfig
from spectrue_core.tools.cache_utils import (
    ensure_diskcache,
    get_cached_context_and_sources,
    make_search_cache_key,
    normalize_domains,
    normalize_exclude_domains,
)
from spectrue_core.tools.search_result_normalizer import (
    build_sources_list,
    clean_tavily_results,
    format_context_from_ranked,
)
from spectrue_core.tools.search_scoring import is_trusted_host, rank_and_filter
from spectrue_core.tools.tavily_client import TavilyClient
from spectrue_core.tools.url_utils import clean_url_for_cache, normalize_host
from spectrue_core.utils.trace import Trace

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)


class WebSearchTool:
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        self.api_key = config.tavily_api_key if config else None

        conc = int(getattr((config.runtime.search if config else None), "tavily_concurrency", 8) or 8)
        global_exclude = []
        try:
            global_exclude = list((config.runtime.search.tavily_exclude_domains or []) if config else [])
        except Exception:
            global_exclude = []

        self._tavily = TavilyClient(
            api_key=self.api_key,
            concurrency=conc,
            global_exclude_domains=global_exclude,
        )
        # Back-compat: older tests patch `tool.client.post`.
        self._client_alias = self._tavily._client

        self.cache: diskcache.Cache = ensure_diskcache(Path("data/cache/web"))
        self.ttl = 86400

        self.page_cache: diskcache.Cache = ensure_diskcache(Path("data/cache/page_text"))
        self.page_ttl = 86400 * 7

        self.last_cache_hit: bool = False
        self.last_search_meta: dict = {}

    @property
    def client(self):
        return self._tavily._client

    @client.setter
    def client(self, value) -> None:
        self._tavily._client = value

    def _normalize_params(self, search_depth: str, max_results: int) -> Tuple[str, int]:
        """
        Validate search parameters against Tavily API limits.
        
        This is a PURE VALIDATOR - it only clamps to API limits (1..10).
        Policy defaults (e.g., 5 for basic, 7 for advanced) belong at the 
        orchestration/caller level, not here.
        """
        depth = (search_depth or "basic").lower()
        if depth not in ("basic", "advanced"):
            depth = "basic"
        # M106: Pure validation - clamp to API limits only
        # Caller is responsible for providing a sensible default
        m = int(max_results) if max_results else 5  # 5 is signature default, not hidden here
        return depth, max(1, min(m, 10))

    def _is_trusted_host(self, host: str) -> bool:
        return is_trusted_host(host)

    def _rank_and_filter(self, query: str, results: list[dict]) -> list[dict]:
        return rank_and_filter(
            query,
            results,
            runtime_config=(self.config.runtime if self.config else None),
        )

    def _clean_results(self, results: list[dict]) -> list[dict]:
        return clean_tavily_results(results)

    def _relevance_score(
        self, query: str, title: str, content: str, url: str, *, tavily_score: float | None = None
    ) -> float:
        from spectrue_core.tools.search_scoring import relevance_score

        return relevance_score(
            query,
            title,
            content,
            url,
            tavily_score=tavily_score,
            runtime_config=(self.config.runtime if self.config else None),
        )

    def _raw_content_mode(self, *, depth: str, domains: list[str] | None) -> bool:
        try:
            forced = self.config.runtime.search.tavily_include_raw_content if self.config else None
        except Exception:
            forced = None
        if forced is True:
            return True
        if forced is False:
            return False
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
        try:
            return bool(self.config.runtime.features.fulltext_fetch if self.config else False)
        except Exception:
            return False

    def _clean_url_key(self, url: str) -> str:
        return clean_url_for_cache(url)

    async def _fetch_extract_text(self, url: str) -> str | None:
        from spectrue_core.tools.url_utils import is_valid_public_http_url

        if not is_valid_public_http_url(url) or not self.api_key:
            return None

        clean_url = self._clean_url_key(url)
        cache_key = f"page_tavily|{clean_url}"
        try:
            if cache_key in self.page_cache:
                cached = self.page_cache[cache_key]
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception:
            pass

        try:
            data = await self._tavily.extract(url=url, format="markdown")
            results = data.get("results", [])
            if not results:
                return None

            extracted_text = ""
            for item in results:
                if item.get("url") == url:
                    extracted_text = item.get("raw_content") or item.get("content") or ""
                    break
            if not extracted_text and results:
                extracted_text = results[0].get("raw_content") or results[0].get("content") or ""
            if not extracted_text:
                return None

            cleaned = ""
            if trafilatura and ("<html" in extracted_text or "<body" in extracted_text or "<div" in extracted_text):
                try:
                    cleaned = trafilatura.extract(
                        extracted_text,
                        include_links=True,
                        include_images=False,
                        include_comments=False,
                    )
                except Exception as e:
                    logger.warning("[Tavily] Trafilatura extraction failed: %s", e)

            if not cleaned:
                cleaned = extracted_text
            cleaned = cleaned.strip()
            if len(cleaned) < 50:
                return None

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

        texts = await asyncio.gather(*[self._fetch_extract_text(u) for u in target_urls])
        url_map = {u: t for u, t in zip(target_urls, texts) if t}

        updated: list[dict] = []
        fetched = 0
        for item in ranked:
            url = item.get("url")
            if url in url_map:
                full = url_map[url]
                new_item = dict(item)
                new_item["content"] = full
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

    async def search(
        self,
        query: str,
        num_results: int = 5,
        depth: str = "basic",
        raw_content: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        topic: str = "general",
    ) -> tuple[str, list[dict]]:
        if not self.api_key:
            logger.debug("[Tavily] API key missing; skipping search")
            return ("", [])

        if not query or not isinstance(query, str) or len(query.strip()) < 3:
            logger.debug("[Tavily] Query too short or empty, skipping search")
            return ("", [])

        _d, _l = self._normalize_params(depth, num_results)
        return await self._search_internal(
            query=query,
            num_results=_l,
            depth=_d,
            raw_content=raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            topic=topic,
        )

    async def _search_internal(
        self,
        *,
        query: str,
        num_results: int,
        depth: str,
        raw_content: bool,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        topic: str,
        ttl: int | None = None,
    ) -> tuple[str, list[dict]]:
        from spectrue_core.utils.text_processing import normalize_search_query

        q = normalize_search_query(query)
        limit = num_results

        normalized_include = normalize_domains(include_domains)
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

        normalized_exclude = normalize_exclude_domains(exclude_domains, cap=32)
        cache_key = make_search_cache_key(
            query=q,
            depth=depth,
            limit=effective_limit,
            raw_mode=raw_mode,
            include_domains=normalized_include,
            exclude_domains=normalized_exclude,
            topic=topic,
        )

        effective_ttl = ttl if ttl is not None else self.ttl
        self.last_cache_hit = False

        if cache_key in self.cache:
            cached = get_cached_context_and_sources(self.cache, cache_key)
            if cached:
                self.last_cache_hit = True
                context_str, sources_list = cached
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
                return context_str, sources_list

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
            logger.debug(
                "[Tavily] Searching: '%s...' (depth=%s, limit=%s%s)",
                q[:100],
                depth,
                effective_limit,
                domain_info,
            )
            response = await self._tavily.search(
                query=q,
                depth=depth,
                max_results=effective_limit,
                include_domains=normalized_include,
                exclude_domains=normalized_exclude,
                topic=topic,
                include_raw_content=raw_mode,  # M106: Pass through raw mode
            )
            results_raw = response.get("results", [])
            logger.debug("[Tavily] Got %d raw results", len(results_raw))

            cleaned = clean_tavily_results(results_raw)
            ranked = self._rank_and_filter(q, cleaned)
            logger.debug("[Tavily] After cleaning: %d results", len(cleaned))
            logger.debug("[Tavily] After ranking/filter: %d results", len(ranked))
            Trace.event(
                "search.results.sample",
                {
                    "query": q,
                    "count": len(ranked),
                    "items": [
                        {
                            "title": r.get("title"),
                            "url": r.get("url"),
                            "relevance_score": r.get("relevance_score"),
                        }
                        for r in (ranked or [])[:5]
                    ],
                },
            )

            def _domain(u: str) -> str:
                try:
                    return normalize_host(urlparse(u).netloc or "")
                except Exception:
                    return ""

            uniq = {_domain(r.get("url", "")) for r in (ranked or []) if r.get("url")}
            uniq.discard("")
            top_relevance = ranked[0].get("relevance_score", 0.0) if ranked else 0.0
            DIVERSIFY_MIN_RELEVANCE = 0.25

            if len(uniq) < 3 and ranked and top_relevance >= DIVERSIFY_MIN_RELEVANCE:
                dom0 = _domain(ranked[0].get("url", ""))
                dom0_normalized = dom0.lower().lstrip(".") if dom0 else ""
                include_set = {d.lower() for d in (normalized_include or [])}
                base_exclude = list(normalized_exclude or [])

                if dom0_normalized and dom0_normalized in include_set:
                    logger.debug(
                        "[Tavily] Diversify pass skipped: dominant domain=%s is in include_domains",
                        dom0_normalized,
                    )
                elif dom0_normalized and dom0_normalized in base_exclude:
                    logger.debug(
                        "[Tavily] Diversify pass skipped: %s already in exclude_domains",
                        dom0_normalized,
                    )
                elif dom0_normalized:
                    logger.debug("[Tavily] Diversify pass: excluding dominant domain=%s", dom0_normalized)
                    try:
                        if len(base_exclude) >= 32:
                            base_exclude = base_exclude[:31]
                        diversify_exclude = base_exclude + [dom0_normalized]

                        response2 = await self._tavily.search(
                            query=q,
                            depth=depth,
                            max_results=effective_limit,
                            include_domains=normalized_include,
                            exclude_domains=diversify_exclude,
                            topic="general",
                            include_raw_content=raw_mode,  # M106: Consistent with main search
                        )
                        results2 = clean_tavily_results(response2.get("results", []))
                        merged = cleaned + results2
                        ranked = self._rank_and_filter(q, merged)
                        logger.debug("[Tavily] Diversify pass done: %d results", len(ranked))
                    except Exception as div_err:
                        logger.debug("[Tavily] Diversify pass failed: %s", div_err)
            elif len(uniq) < 3 and ranked and top_relevance < DIVERSIFY_MIN_RELEVANCE:
                logger.debug(
                    "[Tavily] Diversify pass skipped: top_relevance=%.3f < %.2f (junk results)",
                    top_relevance,
                    DIVERSIFY_MIN_RELEVANCE,
                )

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

            context = format_context_from_ranked(ranked)

            def safe_is_trusted(u: str) -> bool:
                try:
                    return self._is_trusted_host(urlparse(u).netloc)
                except Exception:
                    return False

            sources_list = build_sources_list(ranked, is_trusted_url=safe_is_trusted)

            logger.debug("[Tavily] Generated context: %d chars, %d sources", len(context), len(sources_list))
            if ranked and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Tavily] Sample result: %s...", ranked[0]["title"][:80])

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

            return context, sources_list
        except Exception as e:
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
        await self._tavily.close()
        self.cache.close()
        self.page_cache.close()
