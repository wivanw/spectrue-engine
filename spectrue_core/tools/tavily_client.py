# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import asyncio
import logging
import math
import random

import httpx

from spectrue_core.utils.trace import Trace
from spectrue_core.billing.metering import TavilyMeter

logger = logging.getLogger(__name__)

# Retry configuration for Tavily API errors
TAVILY_MAX_RETRIES = 3
TAVILY_RETRY_BASE_DELAY_S = 0.5  # Base delay in seconds
TAVILY_RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
TAVILY_RETRYABLE_STATUS_CODES = {429, 432, 500, 502, 503, 504}  # Rate limit + server errors


class TavilyClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        timeout_s: float = 12.0,
        concurrency: int = 8,
        global_exclude_domains: list[str] | None = None,
        meter: TavilyMeter | None = None,
    ):
        self.api_key = api_key
        self._sem = asyncio.Semaphore(max(1, min(int(concurrency or 8), 16)))
        self._global_exclude_domains = list(global_exclude_domains or [])
        self._meter = meter

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

    async def _request_with_retry(
        self,
        *,
        url: str,
        payload: dict,
        trace_event_prefix: str,
    ) -> httpx.Response:
        """
        Execute POST request with exponential backoff retry for transient errors.
        
        Retries on: 429 (rate limit), 432 (Tavily internal), 5xx (server errors).
        Uses jitter to prevent thundering herd on concurrent retries.
        """
        last_error: Exception | None = None

        for attempt in range(TAVILY_MAX_RETRIES + 1):
            try:
                Trace.event(f"{trace_event_prefix}.request", {"url": url, "payload": payload})
                r = await self._client.post(url, json=payload)
                r.raise_for_status()
                Trace.event(f"{trace_event_prefix}.response", {"status_code": r.status_code, "text": r.text})
                return r

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code if e.response is not None else 0
                is_retryable = status_code in TAVILY_RETRYABLE_STATUS_CODES

                if is_retryable and attempt < TAVILY_MAX_RETRIES:
                    # Calculate delay with exponential backoff + jitter
                    delay = TAVILY_RETRY_BASE_DELAY_S * (TAVILY_RETRY_BACKOFF_FACTOR ** attempt)
                    jitter = random.uniform(0, delay * 0.3)  # Add up to 30% jitter
                    total_delay = delay + jitter

                    Trace.event(f"{trace_event_prefix}.retry", {
                        "attempt": attempt + 1,
                        "status_code": status_code,
                        "delay_s": round(total_delay, 2),
                        "reason": "retryable_error",
                    })
                    logger.warning(
                        "[Tavily] HTTP %d error, retrying in %.2fs (attempt %d/%d)",
                        status_code, total_delay, attempt + 1, TAVILY_MAX_RETRIES,
                    )
                    await asyncio.sleep(total_delay)
                    last_error = e
                    continue

                # Non-retryable or max retries exceeded
                last_error = e
                break

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                # Network errors are retryable
                if attempt < TAVILY_MAX_RETRIES:
                    delay = TAVILY_RETRY_BASE_DELAY_S * (TAVILY_RETRY_BACKOFF_FACTOR ** attempt)
                    jitter = random.uniform(0, delay * 0.3)
                    total_delay = delay + jitter

                    Trace.event(f"{trace_event_prefix}.retry", {
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "delay_s": round(total_delay, 2),
                        "reason": "network_error",
                    })
                    logger.warning(
                        "[Tavily] Network error (%s), retrying in %.2fs (attempt %d/%d)",
                        type(e).__name__, total_delay, attempt + 1, TAVILY_MAX_RETRIES,
                    )
                    await asyncio.sleep(total_delay)
                    last_error = e
                    continue

                last_error = e
                break

        # All retries exhausted or non-retryable error
        if last_error is not None:
            raise last_error
        # Should not reach here, but just in case
        raise RuntimeError("Unexpected state in _request_with_retry")

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
        include_raw_content: bool = False,  # Allow caller to control raw content
    ) -> dict:
        payload: dict = {"query": query, "search_depth": depth, "max_results": max_results}
        if include_domains:
            payload["include_domains"] = include_domains

        merged_exclude = self._merge_excludes(include_domains=include_domains, exclude_domains=exclude_domains)
        if merged_exclude:
            payload["exclude_domains"] = merged_exclude

        payload["topic"] = topic
        payload["include_raw_content"] = include_raw_content  # Pass through from caller

        url = "https://api.tavily.com/search"
        async with self._sem:
            try:
                r = await self._request_with_retry(
                    url=url,
                    payload=payload,
                    trace_event_prefix="tavily",
                )
                result = r.json()

            except httpx.HTTPStatusError as e:
                # Special handling for 400 Bad Request: retry with minimal payload
                if e.response is not None and e.response.status_code == 400:
                    result = await self._handle_400_with_minimal_payload(
                        url=url,
                        query=query,
                        depth=depth,
                        max_results=max_results,
                        include_domains=include_domains,
                        topic=topic,
                        original_payload=payload,
                        original_error=e,
                    )
                else:
                    raise

            if self._meter:
                try:
                    event = self._meter.record_search(response=result)
                    Trace.event("tavily.metering.recorded", {
                        "type": "search",
                        "cost_credits": str(event.cost_credits),
                    })
                except Exception as exc:
                    Trace.event("tavily.metering.failed", {"error": str(exc)[:200]})
            # Skip tavily.metering.skipped trace - provides no value
            return result

    async def _handle_400_with_minimal_payload(
        self,
        *,
        url: str,
        query: str,
        depth: str,
        max_results: int,
        include_domains: list[str] | None,
        topic: str,
        original_payload: dict,
        original_error: httpx.HTTPStatusError,
    ) -> dict:
        """Handle 400 Bad Request by retrying with minimal payload."""

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
            logger.error("[Tavily] 400 Error Payload: %s", _safe_payload_for_log(original_payload))
            logger.error("[Tavily] 400 Response: %s", (original_error.response.text or "")[:800])
        except Exception:
            pass

        logger.debug(
            "[Tavily] 400 Bad Request. Retrying with minimal payload. Response: %s",
            (original_error.response.text or "")[:500],
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

        try:
            r2 = await self._request_with_retry(
                url=url,
                payload=minimal,
                trace_event_prefix="tavily",
            )
            return r2.json()
        except httpx.HTTPStatusError as e2:
            if e2.response is not None:
                try:
                    logger.error("[Tavily] 400 Error Payload (minimal): %s", _safe_payload_for_log(minimal))
                    logger.error("[Tavily] 400 Response (minimal): %s", (e2.response.text or "")[:800])
                except Exception:
                    pass
            raise

    async def extract(self, *, url: str, format: str = "markdown") -> dict:
        raise RuntimeError(
            "TavilyClient.extract(url=...) is forbidden. "
            "Use extract_batch(urls=[...]) instead."
        )

    async def extract_single_via_batch(self, *, url: str, format: str = "markdown") -> dict:
        return await self.extract_batch(urls=[url], format=format)

    async def extract_batch(self, *, urls: list[str], format: str = "markdown") -> dict:
        """
        Extract content from multiple URLs in a single API call.
        
        Tavily bills 1 credit per 5 URLs for basic extract. If we call extract
        with {"urls":[single]} we burn a full batch for 1/5 of capacity.
        
        This method preserves semantics: same URLs, same content, fewer HTTP calls.
        """
        # Stable dedupe (order-preserving) + strip
        urls2 = list(dict.fromkeys(u.strip() for u in urls if u and u.strip()))
        if not urls2:
            return {"results": []}
        
        Trace.event("tavily.extract.batch", {"urls_count": len(urls2)})
        
        endpoint = "https://api.tavily.com/extract"
        payload = {"urls": urls2, "format": format}
        
        async with self._sem:
            r = await self._request_with_retry(
                url=endpoint,
                payload=payload,
                trace_event_prefix="tavily.extract",
            )
            result = r.json()
            
            if self._meter:
                try:
                    # Tavily pricing: 1 credit per 5 URLs (basic extract)
                    credits_used = float(math.ceil(len(urls2) / 5))
                    event = self._meter.record_extract(
                        response=result,
                        credits_used=credits_used,
                        meta={
                            "urls_count": len(urls2),
                            "batch_size": 5,
                            "credits_used": credits_used,
                        },
                    )
                    Trace.event("tavily.metering.recorded", {
                        "type": "extract_batch",
                        "urls_count": len(urls2),
                        "cost_credits": str(event.cost_credits),
                    })
                except Exception as exc:
                    Trace.event("tavily.metering.failed", {"error": str(exc)[:200]})
            # Skip tavily.metering.skipped trace - provides no value
            
            return result
