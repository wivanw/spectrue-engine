# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest
from unittest.mock import AsyncMock, patch
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.config import SpectrueConfig

@pytest.fixture
def mock_llm_client():
    with patch("spectrue_core.agents.fact_checker_agent.LLMClient") as mock:
        yield mock

@pytest.mark.asyncio
async def test_analyze_calls_llm_client_responses(mock_llm_client):
    """
    Verifies that analyze uses LLMClient.call_json (Responses API) without max_output_tokens for now.
    """
    mock_response = {
        "verified_score": 0.8,
        "context_score": 0.7,
        "style_score": 0.9,
        "rationale": "Test rationale",
        "analysis": "Test analysis"
    }
    
    mock_instance = mock_llm_client.return_value
    mock_instance.call_json = AsyncMock(return_value=mock_response)
    
    config = SpectrueConfig(openai_api_key="sk-test", tavily_api_key="test", openai_model="gpt-5-nano")
    agent = FactCheckerAgent(config)
    
    result = await agent.analyze("Test fact", "Test context", "en", analysis_mode="general")
    
    assert result == mock_response
    
    # Verify call arguments
    mock_instance.call_json.assert_called_once()
    call_kwargs = mock_instance.call_json.call_args.kwargs
    
    assert call_kwargs["model"] == "gpt-5-nano"
    # assert "max_output_tokens" not in call_kwargs  # Should NOT be present as per user request
    if "max_output_tokens" in call_kwargs:
        assert call_kwargs["max_output_tokens"] is None
    # assert "temperature" not in call_kwargs
    if "temperature" in call_kwargs:
        assert call_kwargs["temperature"] is None
    assert call_kwargs["reasoning_effort"] == "low"
    assert call_kwargs["trace_kind"] == "analysis"
    assert "final_analysis_en_v1" in call_kwargs["cache_key"]
