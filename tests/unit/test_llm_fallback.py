# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

"""Unit tests for LLM fallback wrapper."""

import pytest
from unittest.mock import AsyncMock, patch
from spectrue_core.llm.fallback import call_with_fallback
from spectrue_core.llm.failures import classify_llm_failure, LLMFailureKind


class ConnectionErrorStub(Exception):
    pass


class InvalidJsonStub(Exception):
    pass


class SchemaStub(Exception):
    pass


def test_classify_failure():
    """Test standard failure classification."""
    # Timeout
    assert classify_llm_failure(Exception("Request timed out")) == LLMFailureKind.TIMEOUT
    assert classify_llm_failure(Exception("ReadTimeoutError")) == LLMFailureKind.TIMEOUT
    
    # Connection
    assert classify_llm_failure(ConnectionErrorStub("Connection refused")) == LLMFailureKind.CONNECTION_ERROR
    assert classify_llm_failure(Exception("Network unreachable")) == LLMFailureKind.CONNECTION_ERROR
    
    # Provider
    assert classify_llm_failure(Exception("Rate limit exceeded")) == LLMFailureKind.PROVIDER_ERROR
    assert classify_llm_failure(Exception("503 Service Unavailable")) == LLMFailureKind.PROVIDER_ERROR
    
    # JSON
    assert classify_llm_failure(InvalidJsonStub("Expecting value: line 1")) == LLMFailureKind.INVALID_JSON
    assert classify_llm_failure(Exception("JSONDecodeError")) == LLMFailureKind.INVALID_JSON
    
    # Schema
    assert classify_llm_failure(SchemaStub("Validation failed: missing field")) == LLMFailureKind.SCHEMA_VALIDATION_FAILED
    assert classify_llm_failure(Exception("Schema error: unexpected field")) == LLMFailureKind.SCHEMA_VALIDATION_FAILED
    
    # Unknown
    assert classify_llm_failure(Exception("Some business logic error")) is None


@pytest.mark.asyncio
async def test_fallback_primary_success():
    """Test that primary success returns result and doesn't call fallback."""
    primary = AsyncMock(return_value={"status": "ok"})
    fallback = AsyncMock()
    
    result = await call_with_fallback(
        primary_call=primary,
        fallback_call=fallback,
        task_name="test.task",
    )
    
    assert result == {"status": "ok"}
    primary.assert_called_once()
    fallback.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_on_connection_error():
    """Test fallback triggered on connection error."""
    # Simulate connection error
    error = ConnectionErrorStub("Connection refused")
    primary = AsyncMock(side_effect=error)
    fallback = AsyncMock(return_value={"status": "recovered"})
    
    result = await call_with_fallback(
        primary_call=primary,
        fallback_call=fallback,
        task_name="test.task",
    )
    
    assert result == {"status": "recovered"}
    primary.assert_called_once()
    fallback.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_on_invalid_json():
    """Test fallback triggered on JSON parse error."""
    error = InvalidJsonStub("Expecting value: line 1 column 1 (char 0) - JSON parse error")
    primary = AsyncMock(side_effect=error)
    fallback = AsyncMock(return_value={"status": "recovered_json"})
    
    result = await call_with_fallback(
        primary_call=primary,
        fallback_call=fallback,
        task_name="test.task",
    )
    
    assert result == {"status": "recovered_json"}
    primary.assert_called_once()
    fallback.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_both_fail():
    """Test that if both fail, the fallback error is raised."""
    primary_error = ConnectionErrorStub("Primary down")
    fallback_error = ConnectionErrorStub("Fallback down")
    
    primary = AsyncMock(side_effect=primary_error)
    fallback = AsyncMock(side_effect=fallback_error)
    
    with pytest.raises(ConnectionErrorStub) as exc:
        await call_with_fallback(
            primary_call=primary,
            fallback_call=fallback,
            task_name="test.task",
        )
    
    assert "Fallback down" in str(exc.value)
    primary.assert_called_once()
    fallback.assert_called_once()


@pytest.mark.asyncio
@patch("spectrue_core.llm.fallback.Trace")
async def test_fallback_traces(mock_trace):
    """Test that trace events are emitted correctly."""
    error = ConnectionErrorStub("Connection refused")
    primary = AsyncMock(side_effect=error)
    fallback = AsyncMock(return_value={"ok": True})
    
    await call_with_fallback(
        primary_call=primary,
        fallback_call=fallback,
        task_name="test.trace",
        primary_provider_name="p1",
        fallback_provider_name="f1",
    )
    
    # Verify trace calls
    # 1. llm.call.failed
    # 2. llm.fallback.used
    # 3. llm.call.ok (for fallback)
    
    calls = mock_trace.event.call_args_list
    events = [c[0][0] for c in calls]
    
    assert "llm.call.failed" in events
    assert "llm.fallback.used" in events
    assert "llm.call.ok" in events
    
    # Check fallback.used payload
    fallback_used_call = next(c for c in calls if c[0][0] == "llm.fallback.used")
    payload = fallback_used_call[0][1]
    assert payload["task"] == "test.trace"
    assert payload["primary_provider"] == "p1"
    assert payload["fallback_provider"] == "f1"
    assert payload["failure_kind"] == "connection_error"
