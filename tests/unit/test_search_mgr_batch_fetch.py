# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# This file is licensed under the GNU Affero General Public License.

import asyncio
from unittest.mock import AsyncMock

import pytest

from spectrue_core.adapters.retrieval.search_mgr import SearchManager
from spectrue_core.config import SpectrueConfig


@pytest.mark.asyncio
async def test_fetch_urls_content_batch_parallelizes_batches():
    config = SpectrueConfig(tavily_api_key="test-key")
    mgr = SearchManager(config)

    mgr._resolve_extract_batch_parallelism = lambda: 4  # type: ignore[method-assign]
    mgr.web_tool._try_page_cache = lambda url: None
    mgr.web_tool._write_page_cache = lambda url, text: None
    mgr.web_tool._clean_extracted_text = lambda raw: raw

    inflight = {"current": 0, "max": 0}

    async def _extract_batch(*, urls, format="markdown"):
        inflight["current"] += 1
        inflight["max"] = max(inflight["max"], inflight["current"])
        await asyncio.sleep(0.01)
        inflight["current"] -= 1
        return {"results": [{"url": u, "raw_content": f"content:{u}"} for u in urls]}

    mgr.web_tool._tavily.extract_batch = AsyncMock(side_effect=_extract_batch)

    urls = [f"https://example.com/doc-{i}" for i in range(15)]  # 3 batches (5 each)
    result = await mgr.fetch_urls_content_batch(urls)

    assert mgr.web_tool._tavily.extract_batch.await_count == 3
    assert inflight["max"] > 1
    assert len(result) == 15


@pytest.mark.asyncio
async def test_fetch_urls_content_batch_respects_parallelism_limit():
    config = SpectrueConfig(tavily_api_key="test-key")
    mgr = SearchManager(config)

    mgr._resolve_extract_batch_parallelism = lambda: 2  # type: ignore[method-assign]
    mgr.web_tool._try_page_cache = lambda url: None
    mgr.web_tool._write_page_cache = lambda url, text: None
    mgr.web_tool._clean_extracted_text = lambda raw: raw

    inflight = {"current": 0, "max": 0}

    async def _extract_batch(*, urls, format="markdown"):
        inflight["current"] += 1
        inflight["max"] = max(inflight["max"], inflight["current"])
        await asyncio.sleep(0.01)
        inflight["current"] -= 1
        return {"results": [{"url": u, "raw_content": f"content:{u}"} for u in urls]}

    mgr.web_tool._tavily.extract_batch = AsyncMock(side_effect=_extract_batch)

    urls = [f"https://example.com/doc-{i}" for i in range(25)]  # 5 batches
    result = await mgr.fetch_urls_content_batch(urls)

    assert mgr.web_tool._tavily.extract_batch.await_count == 5
    assert inflight["max"] <= 2
    assert len(result) == 25
