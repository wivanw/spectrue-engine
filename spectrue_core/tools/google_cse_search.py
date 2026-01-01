# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import logging
from pathlib import Path

import diskcache
import httpx

from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


class GoogleCSESearchTool:
    """
    Fallback search via Google Custom Search JSON API.
    Enabled only when both key and cx are configured.
    """

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, config: SpectrueConfig):
        self.config = config
        # Allow using a single Google API key for both CSE and Fact Check if desired.
        self.api_key = config.google_search_api_key or config.google_fact_check_key
        self.cse_id = config.google_search_cse_id

        self.client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        cache_dir = Path("data/cache/google_cse")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_dir))
        self.ttl = 86400

        self.last_cache_hit: bool = False

    def enabled(self) -> bool:
        return bool(self.api_key and self.cse_id)

    async def search(
        self,
        query: str,
        *,
        lang: str = "en",
        max_results: int = 5,
        ttl: int | None = None,
    ) -> tuple[str, list[dict]]:
        if not self.enabled():
            return ("", [])

        q = " ".join((query or "").split()).strip()
        if len(q) < 3:
            return ("", [])

        num = max(1, min(int(max_results or 5), 10))
        effective_ttl = ttl if ttl is not None else self.ttl

        cache_key = f"{q}|{lang}|{num}"
        self.last_cache_hit = False

        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if isinstance(cached, tuple) and len(cached) == 2:
                self.last_cache_hit = True
                return cached
            try:
                del self.cache[cache_key]
            except Exception:
                pass

        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": q,
                "num": num,
            }
            # Hint language; Google CSE doesn't guarantee strict language filtering.
            if lang:
                params["hl"] = lang

            Trace.event("google_cse.request", {"url": self.BASE_URL, "params": params})
            resp = await self.client.get(self.BASE_URL, params=params)
            Trace.event(
                "google_cse.response",
                {
                    "status_code": resp.status_code,
                    "text": resp.text,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items") or []
            sources: list[dict] = []
            blocks: list[str] = []

            for it in items[:num]:
                title = (it.get("title") or "").strip()
                link = (it.get("link") or "").strip()
                snippet = (it.get("snippet") or "").strip()
                if not link:
                    continue
                if not title:
                    title = link
                sources.append(
                    {
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "origin": "WEB",
                        "provider": "google_cse",
                    }
                )
                blocks.append(f"Source: {title}\nURL: {link}\nContent: {snippet}\n---")

            context = "\n".join(blocks)
            self.cache.set(cache_key, (context, sources), expire=effective_ttl)
            return (context, sources)
        except Exception as e:
            logger.warning("[Google CSE] search failed: %s", e)
            return ("", [])