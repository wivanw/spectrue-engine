# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.tools.web_search_tool import WebSearchTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig


@pytest.fixture
def mock_config():
    """Provides a dummy configuration."""
    config = MagicMock(spec=SpectrueConfig)
    config.tavily_api_key = "test-tavily-key"
    config.google_search_api_key = "test-google-key"
    config.google_search_cse_id = "test-cse-id"
    config.google_fact_check_key = "test-fc-key"
    
    # Mock runtime config
    config.runtime = MagicMock()
    config.runtime.search = MagicMock()
    config.runtime.search.tavily_concurrency = 1
    config.runtime.search.tavily_exclude_domains = []
    config.runtime.features = MagicMock()
    config.runtime.features.fulltext_fetch = False
    
    # ClaimGraph config (disabled by default in tests)
    config.runtime.claim_graph = MagicMock()
    config.runtime.claim_graph.enabled = False
    
    return config

@pytest.fixture
def mock_llm_client():
    """Matches the interface of LLMClient, returning AsyncMocks."""
    client = MagicMock(spec=LLMClient)
    client.call = AsyncMock(return_value={
        "content": "Mocked LLM content",
        "parsed": None,
        "model": "gpt-5-nano",
        "cache_status": "NONE",
        "usage": {"total_tokens": 100}
    })
    client.call_json = AsyncMock(return_value={
        "mock_key": "mock_value"
    })
    client.close = AsyncMock()
    return client

@pytest.fixture
def mock_web_search_tool():
    """Matches the interface of WebSearchTool."""
    tool = MagicMock(spec=WebSearchTool)
    tool.search = AsyncMock(return_value=(
        "Mock Search Context",
        [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1"},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2"},
        ]
    ))
    return tool

@pytest.fixture
def mock_cse_tool():
    """Matches the interface of GoogleCSESearchTool."""
    tool = MagicMock(spec=GoogleCSESearchTool)
    tool.enabled.return_value = True
    tool.search = AsyncMock(return_value=(
        "Mock CSE Context",
        [
            {"title": "CSE Result 1", "link": "https://google.com/1", "snippet": "Snippet 1"},
        ]
    ))
    return tool

@pytest.fixture
def mock_httpx_client():
    """Mocks httpx.AsyncClient for testing tools internals."""
    client = MagicMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.aclose = AsyncMock()
    
    # Setup standard response structure
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {}
    mock_response.text = ""
    
    client.post.return_value = mock_response
    client.get.return_value = mock_response
    
    return client
