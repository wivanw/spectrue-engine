# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for TavilyClient retry logic with exponential backoff."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from spectrue_core.tools.tavily_client import (
    TavilyClient,
    TAVILY_MAX_RETRIES,
    TAVILY_RETRYABLE_STATUS_CODES,
)


@pytest.fixture
def tavily_client():
    """Create a TavilyClient instance for testing."""
    return TavilyClient(api_key="test-api-key", timeout_s=5.0)


@pytest.fixture
def mock_response_ok():
    """Create a successful mock response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.text = '{"results": []}'
    response.json.return_value = {"results": []}
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_response_432():
    """Create a 432 error response (Tavily internal/rate limit)."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 432
    response.text = "Rate limit exceeded"
    error = httpx.HTTPStatusError("432 error", request=MagicMock(), response=response)
    response.raise_for_status = MagicMock(side_effect=error)
    return response, error


@pytest.fixture  
def mock_response_429():
    """Create a 429 rate limit response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 429
    response.text = "Too Many Requests"
    error = httpx.HTTPStatusError("429 error", request=MagicMock(), response=response)
    response.raise_for_status = MagicMock(side_effect=error)
    return response, error


@pytest.fixture
def mock_response_401():
    """Create a 401 unauthorized response (non-retryable)."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 401
    response.text = "Unauthorized"
    error = httpx.HTTPStatusError("401 error", request=MagicMock(), response=response)
    response.raise_for_status = MagicMock(side_effect=error)
    return response, error


class TestTavilyRetryConstants:
    """Test retry configuration constants."""
    
    def test_max_retries_is_reasonable(self):
        assert TAVILY_MAX_RETRIES >= 1
        assert TAVILY_MAX_RETRIES <= 5
        
    def test_retryable_status_codes_include_rate_limit(self):
        assert 429 in TAVILY_RETRYABLE_STATUS_CODES
        assert 432 in TAVILY_RETRYABLE_STATUS_CODES
        
    def test_retryable_status_codes_include_server_errors(self):
        assert 500 in TAVILY_RETRYABLE_STATUS_CODES
        assert 502 in TAVILY_RETRYABLE_STATUS_CODES
        assert 503 in TAVILY_RETRYABLE_STATUS_CODES
        assert 504 in TAVILY_RETRYABLE_STATUS_CODES
        
    def test_non_retryable_codes_not_included(self):
        # 400 and 401 should NOT be in retryable (400 has special handling)
        assert 400 not in TAVILY_RETRYABLE_STATUS_CODES
        assert 401 not in TAVILY_RETRYABLE_STATUS_CODES
        assert 403 not in TAVILY_RETRYABLE_STATUS_CODES


@pytest.mark.asyncio
class TestTavilyClientRetry:
    """Test TavilyClient retry behavior."""
    
    async def test_successful_request_no_retry(self, tavily_client, mock_response_ok):
        """Successful request should not trigger retries."""
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response_ok
            
            result = await tavily_client.search(
                query="test query",
                depth="basic",
                max_results=3,
            )
            
            assert result == {"results": []}
            assert mock_post.call_count == 1
    
    async def test_retry_on_432_then_success(self, tavily_client, mock_response_432, mock_response_ok):
        """432 error should trigger retry, then succeed."""
        response_432, _ = mock_response_432
        
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            # First call fails with 432, second succeeds
            mock_post.side_effect = [response_432, mock_response_ok]
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await tavily_client.search(
                    query="test query",
                    depth="basic",
                    max_results=3,
                )
                
                assert result == {"results": []}
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
    
    async def test_retry_on_429_with_backoff(self, tavily_client, mock_response_429, mock_response_ok):
        """429 error should trigger retry with exponential backoff."""
        response_429, _ = mock_response_429
        
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            # Two failures, then success
            mock_post.side_effect = [response_429, response_429, mock_response_ok]
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await tavily_client.search(
                    query="test query",
                    depth="basic",
                    max_results=3,
                )
                
                assert result == {"results": []}
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2
                
                # Verify backoff: second delay should be larger than first
                first_delay = mock_sleep.call_args_list[0][0][0]
                second_delay = mock_sleep.call_args_list[1][0][0]
                assert second_delay > first_delay
    
    async def test_max_retries_exceeded(self, tavily_client, mock_response_432):
        """Should raise after max retries exceeded."""
        response_432, error_432 = mock_response_432
        
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            # All calls fail
            mock_post.return_value = response_432
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.HTTPStatusError):
                    await tavily_client.search(
                        query="test query",
                        depth="basic",
                        max_results=3,
                    )
                
                # Should have tried initial + retries
                assert mock_post.call_count == TAVILY_MAX_RETRIES + 1
    
    async def test_no_retry_on_401(self, tavily_client, mock_response_401):
        """401 error should NOT trigger retry."""
        response_401, _ = mock_response_401
        
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response_401
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(httpx.HTTPStatusError):
                    await tavily_client.search(
                        query="test query",
                        depth="basic",
                        max_results=3,
                    )
                
                # Should only try once (no retries for 401)
                assert mock_post.call_count == 1
                mock_sleep.assert_not_called()
    
    async def test_extract_also_retries(self, tavily_client, mock_response_432, mock_response_ok):
        """Extract method should also use retry logic."""
        response_432, _ = mock_response_432
        
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [response_432, mock_response_ok]
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await tavily_client.extract(url="https://example.com")
                
                assert result == {"results": []}
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
    
    async def test_retry_on_timeout(self, tavily_client, mock_response_ok):
        """Timeout errors should trigger retry."""
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            # First call times out, second succeeds
            mock_post.side_effect = [
                httpx.TimeoutException("Timeout"),
                mock_response_ok,
            ]
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await tavily_client.search(
                    query="test query",
                    depth="basic",
                    max_results=3,
                )
                
                assert result == {"results": []}
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
    
    async def test_retry_on_connect_error(self, tavily_client, mock_response_ok):
        """Connection errors should trigger retry."""
        with patch.object(tavily_client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                httpx.ConnectError("Connection refused"),
                mock_response_ok,
            ]
            
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await tavily_client.search(
                    query="test query",
                    depth="basic",
                    max_results=3,
                )
                
                assert result == {"results": []}
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1