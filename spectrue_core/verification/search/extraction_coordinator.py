# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Centralized URL extraction coordinator for batch processing.

This module provides a coordinator pattern for collecting URLs across
multiple code paths and executing batch extraction efficiently.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from spectrue_core.utils.trace import Trace

if TYPE_CHECKING:
    from spectrue_core.verification.search.search_mgr import SearchManager

logger = logging.getLogger(__name__)


class ExtractionCoordinator:
    """
    Centralized URL extraction with deferred batching.
    
    Tavily bills 1 credit per 5 URLs for extraction. This coordinator
    collects URLs from multiple sources, then executes a single batch
    extraction to populate the cache.
    
    Usage:
        coord = ExtractionCoordinator(search_mgr)
        
        # Phase 1: Collect all URLs
        coord.register(url1)
        coord.register(url2)
        ...
        
        # Phase 2: Execute batch extraction (populates cache)
        await coord.execute()
        
        # Phase 3: Downstream code uses cache hits
        # EAL, _enrich_with_fulltext, etc. now get cache hits
    """
    
    def __init__(
        self,
        search_mgr: "SearchManager",
        *,
        batch_size: int = 5,
    ):
        """
        Initialize coordinator.
        
        Args:
            search_mgr: SearchManager with fetch_urls_content_batch method
            batch_size: URLs per batch (Tavily optimal = 5)
        """
        self.search_mgr = search_mgr
        self.batch_size = batch_size
        self._pending: list[str] = []
        self._results: dict[str, str] = {}
        self._executed: bool = False
    
    def register(self, url: str) -> None:
        """
        Register URL for batch extraction.
        
        Deduplicates and preserves order.
        """
        if not url or not isinstance(url, str):
            return
        url = url.strip()
        if url and url not in self._pending and url not in self._results:
            self._pending.append(url)
    
    def register_many(self, urls: list[str]) -> None:
        """Register multiple URLs at once."""
        for url in urls:
            self.register(url)
    
    async def execute(self) -> int:
        """
        Execute batch extraction for all registered URLs.
        
        Populates the search_mgr's internal cache so subsequent
        EAL/enrichment calls get cache hits instead of API calls.
        
        Returns:
            Number of URLs successfully extracted
        """
        if not self._pending:
            return 0
        
        if self._executed:
            logger.warning("[ExtractionCoordinator] Already executed, skipping")
            return len(self._results)
        
        total_urls = len(self._pending)
        Trace.event("extraction.coordinator.execute.start", {
            "total_urls": total_urls,
            "batch_size": self.batch_size,
        })
        
        try:
            # Use search_mgr's batch fetch (handles chunking internally)
            if hasattr(self.search_mgr, "fetch_urls_content_batch"):
                self._results = await self.search_mgr.fetch_urls_content_batch(
                    self._pending
                )
            else:
                logger.warning("[ExtractionCoordinator] search_mgr missing fetch_urls_content_batch")
                self._results = {}
            
            extracted_count = len(self._results)
            Trace.event("extraction.coordinator.execute.done", {
                "total_urls": total_urls,
                "extracted": extracted_count,
                "cache_populated": extracted_count,
            })
            
            self._pending.clear()
            self._executed = True
            return extracted_count
            
        except Exception as e:
            logger.error("[ExtractionCoordinator] Batch extraction failed: %s", e)
            Trace.event("extraction.coordinator.execute.error", {
                "total_urls": total_urls,
                "error": str(e)[:200],
            })
            self._executed = True
            return 0
    
    def get(self, url: str) -> str | None:
        """
        Get extracted content for URL.
        
        Returns content from batch results. Should only be called
        after execute().
        """
        return self._results.get(url)
    
    @property
    def pending_count(self) -> int:
        """Number of URLs pending extraction."""
        return len(self._pending)
    
    @property
    def extracted_count(self) -> int:
        """Number of URLs successfully extracted."""
        return len(self._results)
    
    def reset(self) -> None:
        """Reset coordinator for reuse."""
        self._pending.clear()
        self._results.clear()
        self._executed = False


def collect_urls_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    """
    Extract URLs from a list of source dicts.
    
    Handles both "url" and "link" field names.
    """
    urls: list[str] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        url = src.get("url") or src.get("link")
        if url and isinstance(url, str):
            url = url.strip()
            if url and url not in urls:
                urls.append(url)
    return urls
