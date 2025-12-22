from __future__ import annotations

import asyncio
import logging

import httpx

from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


class TavilyClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        timeout_s: float = 12.0,
        concurrency: int = 8,
        global_exclude_domains: list[str] | None = None,
    ):
        self.api_key = api_key
        self._sem = asyncio.Semaphore(max(1, min(int(concurrency or 8), 16)))
        self._global_exclude_domains = list(global_exclude_domains or [])

        self._client = httpx.AsyncClient(
            timeout=float(timeout_s),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
                "X-Client-Source": "spectrue",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _merge_excludes(
        self,
        *,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
    ) -> list[str] | None:
        merged_exclude = set(self._global_exclude_domains)
        if exclude_domains:
            merged_exclude.update(exclude_domains)

        final_exclude = sorted({d.lower().lstrip(".") for d in merged_exclude if d})
        if not final_exclude:
            return None

        include_set = {d.lower().lstrip(".") for d in (include_domains or [])}
        filtered = [d for d in final_exclude if d not in include_set]
        if len(filtered) > 32:
            filtered = filtered[:32]
        return filtered or None

    async def search(
        self,
        *,
        query: str,
        depth: str,
        max_results: int,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        topic: str = "general",
    ) -> dict:
        payload: dict = {"query": query, "search_depth": depth, "max_results": max_results}
        if include_domains:
            payload["include_domains"] = include_domains

        merged_exclude = self._merge_excludes(include_domains=include_domains, exclude_domains=exclude_domains)
        if merged_exclude:
            payload["exclude_domains"] = merged_exclude

        payload["topic"] = topic
        payload["include_raw_content"] = False

        url = "https://api.tavily.com/search"
        async with self._sem:
            Trace.event("tavily.request", {"url": url, "payload": payload})
            r = await self._client.post(url, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
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
                        pass

                    logger.debug(
                        "[Tavily] 400 Bad Request. Retrying with minimal payload. Response: %s",
                        (e.response.text or "")[:500],
                    )
                    minimal: dict = {
                        "query": query,
                        "search_depth": depth,
                        "max_results": max_results,
                    }
                    if include_domains:
                        minimal["include_domains"] = include_domains
                    retry_exclude = self._merge_excludes(include_domains=include_domains, exclude_domains=None)
                    if retry_exclude:
                        minimal["exclude_domains"] = retry_exclude
                    minimal["topic"] = topic
                    minimal["include_raw_content"] = False

                    Trace.event("tavily.request", {"url": url, "payload": minimal})
                    r2 = await self._client.post(url, json=minimal)
                    try:
                        r2.raise_for_status()
                    except httpx.HTTPStatusError as e2:
                        if e2.response is not None:
                            try:
                                logger.error("[Tavily] 400 Error Payload (minimal): %s", _safe_payload_for_log(minimal))
                                logger.error("[Tavily] 400 Response (minimal): %s", (e2.response.text or "")[:800])
                            except Exception:
                                pass
                        raise
                    Trace.event("tavily.response", {"status_code": r2.status_code, "text": r2.text})
                    return r2.json()

                if e.response is not None:
                    logger.debug(
                        "[Tavily] HTTP error %s. Response: %s",
                        e.response.status_code,
                        (e.response.text or "")[:500],
                    )
                raise

            Trace.event("tavily.response", {"status_code": r.status_code, "text": r.text})
            return r.json()

    async def extract(self, *, url: str, format: str = "markdown") -> dict:
        endpoint = "https://api.tavily.com/extract"
        payload = {"urls": [url], "format": format}
        async with self._sem:
            Trace.event("tavily.extract.request", {"url": endpoint, "target": url})
            r = await self._client.post(endpoint, json=payload)
            r.raise_for_status()
            return r.json()

