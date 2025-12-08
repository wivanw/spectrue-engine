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
from pathlib import Path
import diskcache
from typing import Tuple
from urllib.parse import urlparse
from spectrue_core.config import SpectrueConfig

# M29: Import from centralized trusted sources registry
from spectrue_core.verification.trusted_sources import ALL_TRUSTED_DOMAINS as TRUSTED_DOMAINS


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

        self.client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)

        cache_dir = Path("data/cache/web")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_dir))
        self.ttl = 86400

        try:
            conc = int(os.getenv("TAVILY_CONCURRENCY", "8"))
        except Exception:
            conc = 8
        self._sem = asyncio.Semaphore(max(1, min(conc, 16)))
        self.last_cache_hit: bool = False

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

            cleaned.append({"title": title, "url": url, "content": content})
        return cleaned

    async def _request_search(self, q: str, depth: str, limit: int, domains: list[str] | None = None) -> dict:
        payload = {
            "api_key": self.api_key,
            "query": q,
            "search_depth": depth,
            "max_results": limit
        }
        # M20: Add trusted domains filter for Tier 1 searches
        if domains:
            payload["include_domains"] = domains
        
        url = "https://api.tavily.com/search"
        async with self._sem:
            r = await self.client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    async def search(self, query: str, search_depth: str = "advanced", max_results: int = 10,
                     ttl: int | None = None, domains: list[str] | None = None) -> tuple[str, list[dict]]:
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
        if not query or not isinstance(query, str) or len(query.strip()) < 3:
            print("[Tavily] Query too short or empty, skipping search")
            return ("", [])

        q = " ".join(query.split()).strip()
        depth, limit = self._normalize_params(search_depth, max_results)
        cache_key = f"{q}|{depth}|{limit}"

        effective_ttl = ttl if ttl is not None else self.ttl
        self.last_cache_hit = False

        if cache_key in self.cache:
            self.last_cache_hit = True
            cached_data = self.cache[cache_key]
            # Cache stores tuple (context_str, sources_list)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                context_str, sources_list = cached_data
                print(f"[Tavily] ✓ Cache hit for '{q[:100]}...' ({len(context_str)} chars, {len(sources_list)} sources)")
                return (context_str, sources_list)
            else:
                # Old cache format (string only), invalidate
                print("[Tavily] Cache format outdated, re-fetching")
                del self.cache[cache_key]

        try:
            domain_info = f", domains={len(domains)}" if domains else ""
            print(f"[Tavily] Searching: '{q[:100]}...' (depth={depth}, limit={limit}{domain_info})")
            response = await self._request_search(q, depth, limit, domains)
            results_raw = response.get('results', [])
            print(f"[Tavily] Got {len(results_raw)} raw results")
            
            results = self._clean_results(results_raw)
            print(f"[Tavily] After cleaning: {len(results)} results")
            
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
                return f"Source: {trust_tag}{obj['title']}\nURL: {obj['url']}\nContent: {obj['content']}\n---"
            
            context = "\n".join([format_source(obj) for obj in results])
            
            # Create structured sources for frontend
            sources_list = [
                {
                    "title": obj["title"],
                    "link": obj["url"],
                    "snippet": obj["content"],
                    "origin": "WEB"
                }
                for obj in results
            ]
            
            print(f"[Tavily] Generated context: {len(context)} chars, {len(sources_list)} sources")
            if results:
                print(f"[Tavily] Sample result: {results[0]['title'][:80]}...")
            
            # Cache both context and sources
            if context:
                self.cache.set(cache_key, (context, sources_list), expire=effective_ttl)
            
            return (context, sources_list)
        except Exception as e:
            print(f"[Tavily] ✗ Error during search ({depth}): {e}")
            return ("Ошибка при выполнении поиска в интернете.", [])
