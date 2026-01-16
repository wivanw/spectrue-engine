# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""Tests for Tavily Extract batching optimization."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    def _make_response(results: list[dict]):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"results": results}
        return response
    return _make_response


class TestTavilyClientExtractBatch:
    """Tests for TavilyClient.extract_batch method."""

    @pytest.mark.asyncio
    async def test_extract_batch_single_call_for_3_urls(self, mock_httpx_response):
        """3 URLs should result in 1 HTTP call with all 3 URLs in payload."""
        from spectrue_core.tools.tavily_client import TavilyClient

        client = TavilyClient(api_key="test-key")
        
        # Mock the HTTP client
        mock_post = AsyncMock(return_value=mock_httpx_response([
            {"url": "https://a.com", "content": "Content A"},
            {"url": "https://b.com", "content": "Content B"},
            {"url": "https://c.com", "content": "Content C"},
        ]))
        client._client.post = mock_post

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        await client.extract_batch(urls=urls)

        # Should be 1 HTTP call
        assert mock_post.call_count == 1
        
        # Payload should contain all 3 URLs
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert len(payload["urls"]) == 3
        assert set(payload["urls"]) == {"https://a.com", "https://b.com", "https://c.com"}
        
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_batch_deduplicates_urls(self, mock_httpx_response):
        """Duplicate URLs should be deduplicated before sending."""
        from spectrue_core.tools.tavily_client import TavilyClient

        client = TavilyClient(api_key="test-key")
        mock_post = AsyncMock(return_value=mock_httpx_response([]))
        client._client.post = mock_post

        urls = ["https://a.com", "https://a.com", "https://b.com", " https://b.com "]
        await client.extract_batch(urls=urls)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        # Should only have 2 unique URLs
        assert len(payload["urls"]) == 2
        
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_batch_empty_urls_returns_empty(self):
        """Empty URL list should return empty results without HTTP call."""
        from spectrue_core.tools.tavily_client import TavilyClient

        client = TavilyClient(api_key="test-key")
        mock_post = AsyncMock()
        client._client.post = mock_post

        result = await client.extract_batch(urls=[])
        
        assert result == {"results": []}
        assert mock_post.call_count == 0
        
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_batch_metering_credits_calculation(self, mock_httpx_response):
        """Credits should be calculated as ceil(urls/5)."""
        from spectrue_core.tools.tavily_client import TavilyClient
        from spectrue_core.billing.metering import TavilyMeter
        from spectrue_core.billing.cost_ledger import CostLedger
        from spectrue_core.billing.config_loader import load_pricing_policy

        ledger = CostLedger(run_id="test")
        policy = load_pricing_policy()
        meter = TavilyMeter(ledger=ledger, policy=policy)

        client = TavilyClient(api_key="test-key", meter=meter)
        mock_post = AsyncMock(return_value=mock_httpx_response([]))
        client._client.post = mock_post

        # 3 URLs = ceil(3/5) = 1 credit
        await client.extract_batch(urls=["https://a.com", "https://b.com", "https://c.com"])
        
        # Verify metering was called
        assert len(ledger.events) == 1
        event = ledger.events[0]
        assert event.meta.get("urls_count") == 3
        assert event.meta.get("credits_used") == 1.0
        
        await client.close()


class TestWebSearchToolBatchEnrichment:
    """Tests for WebSearchTool._enrich_with_fulltext batching."""

    @pytest.mark.asyncio
    async def test_enrich_uses_batch_for_missing_urls(self):
        """Missing URLs should be fetched in a single batch call."""
        from spectrue_core.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool(config=None)
        tool.api_key = "test-key"
        
        # Mock page_cache to return no hits
        tool.page_cache = MagicMock()
        tool.page_cache.__contains__ = MagicMock(return_value=False)
        tool.page_cache.set = MagicMock()

        # Mock extract_batch
        tool._tavily.extract_batch = AsyncMock(return_value={
            "results": [
                {"url": "https://a.com", "content": "Content A with enough characters to pass validation threshold."},
                {"url": "https://b.com", "content": "Content B with enough characters to pass validation threshold."},
                {"url": "https://c.com", "content": "Content C with enough characters to pass validation threshold."},
            ]
        })

        ranked = [
            {"url": "https://a.com", "title": "A", "content": "short", "score": 0.9},
            {"url": "https://b.com", "title": "B", "content": "short", "score": 0.8},
            {"url": "https://c.com", "title": "C", "content": "short", "score": 0.7},
        ]

        updated, fetched = await tool._enrich_with_fulltext("test query", ranked, limit=3)

        # Should call extract_batch once with 3 URLs
        assert tool._tavily.extract_batch.call_count == 1
        call_kwargs = tool._tavily.extract_batch.call_args.kwargs
        assert len(call_kwargs["urls"]) == 3
        
        await tool.close()

    @pytest.mark.asyncio
    async def test_enrich_skips_cached_urls(self):
        """Cached URLs should not be included in batch request."""
        from spectrue_core.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool(config=None)
        tool.api_key = "test-key"

        # Mock cache: a.com and b.com are cached
        def cache_contains(key):
            return key in {"page_tavily|https://a.com", "page_tavily|https://b.com"}
        
        def cache_get(key):
            if key == "page_tavily|https://a.com":
                return "Cached content A with enough characters."
            if key == "page_tavily|https://b.com":
                return "Cached content B with enough characters."
            raise KeyError(key)

        tool.page_cache = MagicMock()
        tool.page_cache.__contains__ = MagicMock(side_effect=cache_contains)
        tool.page_cache.__getitem__ = MagicMock(side_effect=cache_get)
        tool.page_cache.set = MagicMock()

        # Mock extract_batch
        tool._tavily.extract_batch = AsyncMock(return_value={
            "results": [
                {"url": "https://c.com", "content": "Content C with enough characters to pass validation."},
            ]
        })

        ranked = [
            {"url": "https://a.com", "title": "A", "content": "short", "score": 0.9},
            {"url": "https://b.com", "title": "B", "content": "short", "score": 0.8},
            {"url": "https://c.com", "title": "C", "content": "short", "score": 0.7},
        ]

        await tool._enrich_with_fulltext("test query", ranked, limit=3)

        # Should call extract_batch with only 1 URL (c.com)
        assert tool._tavily.extract_batch.call_count == 1
        call_kwargs = tool._tavily.extract_batch.call_args.kwargs
        assert call_kwargs["urls"] == ["https://c.com"]
        
        await tool.close()

    @pytest.mark.asyncio
    async def test_enrich_chunks_into_batches_of_5(self):
        """6 URLs should result in 2 batch calls (5 + 1)."""
        from spectrue_core.tools.web_search_tool import WebSearchTool

        tool = WebSearchTool(config=None)
        tool.api_key = "test-key"

        # Mock page_cache to return no hits
        tool.page_cache = MagicMock()
        tool.page_cache.__contains__ = MagicMock(return_value=False)
        tool.page_cache.set = MagicMock()

        # Mock extract_batch to return content for all URLs
        tool._tavily.extract_batch = AsyncMock(return_value={
            "results": [{"url": f"https://{i}.com", "content": f"Content {i} with enough chars to pass validation threshold."} for i in range(6)]
        })

        ranked = [
            {"url": f"https://{i}.com", "title": f"Title {i}", "content": "short", "score": 0.9 - i * 0.1}
            for i in range(6)
        ]

        await tool._enrich_with_fulltext("test query", ranked, limit=6)

        # Should call extract_batch twice (5 + 1)
        assert tool._tavily.extract_batch.call_count == 2
        
        # First batch should have 5 URLs
        first_call = tool._tavily.extract_batch.call_args_list[0].kwargs
        assert len(first_call["urls"]) == 5
        
        # Second batch should have 1 URL
        second_call = tool._tavily.extract_batch.call_args_list[1].kwargs
        assert len(second_call["urls"]) == 1
        
        await tool.close()
