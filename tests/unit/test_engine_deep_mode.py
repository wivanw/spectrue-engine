# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spectrue_core.engine import SpectrueEngine


@pytest.mark.asyncio
async def test_deep_mode_single_pipeline_run_multi_claims(mock_config):
    mock_config.openai_model = "gpt-5-nano"
    mock_config.runtime.tunables = MagicMock()
    mock_config.runtime.tunables.langdetect_min_prob = 0.0
    mock_config.runtime.features.trace_enabled = False
    mock_config.runtime.features.log_redaction = False
    mock_config.runtime.features.trace_safe_payloads = True
    mock_config.runtime.debug = MagicMock()
    mock_config.runtime.debug.trace_max_head_chars = 120
    mock_config.runtime.debug.trace_max_inline_chars = 600

    claims = [
        {"id": "c1", "text": "Claim 1"},
        {"id": "c2", "text": "Claim 2"},
        {"id": "c3", "text": "Claim 3"},
    ]

    initial_result = {
        "_extracted_claims": claims,
        "cost_summary": {"total_credits": 1.0},
    }
    verification_result = {
        "verified_score": 0.6,
        "rgba": [0.0, 0.6, 0.5, 0.7],
        "danger_score": 0.1,
        "claim_verdicts": [
            {"claim_id": "c1", "verdict_score": 0.6},
            {"claim_id": "c2", "verdict_score": 0.4},
            {"claim_id": "c3", "verdict_score": 0.7},
        ],
        "sources": [],
        "cost_summary": {"total_credits": 2.0},
    }

    mock_verifier = MagicMock()
    mock_verifier.verify_fact = AsyncMock(side_effect=[initial_result, verification_result])
    mock_verifier.fetch_url_content = AsyncMock(return_value=None)
    mock_verifier.pipeline = MagicMock()
    mock_verifier.pipeline.search_mgr = MagicMock()
    mock_verifier.pipeline.search_mgr.web_tool = MagicMock()
    mock_verifier.pipeline.search_mgr.web_tool._tavily = MagicMock()
    mock_verifier.pipeline.search_mgr.web_tool._tavily._meter = MagicMock()

    with patch("spectrue_core.engine.FactVerifier", return_value=mock_verifier), \
        patch("spectrue_core.engine.detect_content_language", return_value=("en", 1.0)), \
        patch("spectrue_core.engine.load_pricing_policy", return_value=MagicMock()):
        engine = SpectrueEngine(mock_config)
        result = await engine.analyze_text(
            "Claim 1. Claim 2. Claim 3.",
            lang="en",
            analysis_mode="deep",
        )

    assert mock_verifier.verify_fact.call_count == 2

    first_call = mock_verifier.verify_fact.call_args_list[0].kwargs
    second_call = mock_verifier.verify_fact.call_args_list[1].kwargs

    assert first_call.get("extract_claims_only") is True
    assert second_call.get("preloaded_claims") == claims
    assert second_call.get("pipeline_profile") == "deep"

    assert result["analysis_mode"] == "deep"
    assert result["claims"] == ["Claim 1", "Claim 2", "Claim 3"]
    assert result["judge_mode"] == "deep"
    assert "deep_analysis" in result
    assert "verified_score" not in result
