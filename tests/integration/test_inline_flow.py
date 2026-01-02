# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
from unittest.mock import MagicMock, AsyncMock

from spectrue_core.verification.pipeline.pipeline_search import run_search_flow, SearchFlowInput, SearchFlowState
from spectrue_core.verification.search.search_mgr import SearchManager
from spectrue_core.utils.embedding_service import EmbedService

@pytest.mark.asyncio
async def test_inline_sources_shortcut_flow():
    """
    Integration test verifying that:
    1. Pipeline accepts inline sources
    2. PhaseRunner uses them for verdict (shortcut)
    3. Stance is auto-set to SUPPORT for high similarity
    """
    if not EmbedService.is_available():
        pytest.skip("Embeddings not available")

    # Setup Mocks
    mock_config = MagicMock()
    mock_config.runtime.features.claim_orchestration = True
    # Ensure locale config doesn't return a Mock
    mock_config.runtime.locale = None
    
    mock_search_mgr = MagicMock(spec=SearchManager)
    # Mock calculate_cost to allow search (even if shortcut skips it)
    mock_search_mgr.calculate_cost.return_value = 0
    mock_search_mgr.can_afford.return_value = True
    # M109: PhaseRunner expects (context, results) tuple from search_phase
    mock_search_mgr.search_phase.return_value = ("", [])
    # M109: Also mock ladder to return input list
    mock_search_mgr.apply_evidence_acquisition_ladder.side_effect = lambda x, **kwargs: x
    
    mock_agent = AsyncMock()
    mock_agent.verify_inline_source_relevance.return_value = {
        "is_relevant": True,
        "relevance_score": 0.9,
        "quote_matches": ["PSR J2322-2650b is indeed an exoplanet"],
        "stance": "SUPPORT",
        "is_trusted": True
    }
    
    # Setup Input
    claim_text = "The exoplanet PSR J2322-2650b orbits a millisecond pulsar."
    
    # Inline source that strongly matches the claim
    inline_src = {
        "url": "https://nasa.gov/pulsar-planet",
        "title": "Pulsar Planet Discovery",
        "content": "Astronomers confirm that PSR J2322-2650b is indeed an exoplanet orbiting a millisecond pulsar. The planet has a lemon-like shape.",
        "snippet": "PSR J2322-2650b is indeed an exoplanet orbiting a millisecond pulsar.",
        "is_primary": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "is_trusted": True
    }
    
    inp = SearchFlowInput(
        fact=claim_text,
        lang="en",
        gpt_model="gpt-4o",
        search_type="deep_research",
        max_cost=100,
        article_intent="news",
        search_queries=[claim_text],
        claims=[{"id": "c1", "text": claim_text, "normalized_text": claim_text}],
        preloaded_context=None,
        progress_callback=None,
        inline_sources=[inline_src]  # This is what we fixed!
    )
    
    state = SearchFlowState(
        final_context="",
        final_sources=[],
        preloaded_context=None,
        used_orchestration=False
    )
    
    # Run
    result_state = await run_search_flow(
        config=mock_config,
        search_mgr=mock_search_mgr,
        agent=mock_agent,
        can_add_search=lambda *args: True,
        inp=inp,
        state=state
    )
    
    # Verification
    # 1. Orchestration should have been used
    assert result_state.used_orchestration is True, "Orchestration should be enabled"
    
    # 2. Execution state should show sufficiency for c1
    exec_state = result_state.execution_state
    c1_state = exec_state.get("c1", {})
    
    assert c1_state.get("is_sufficient") is True, f"Claim should be sufficient via inline shortcut. State: {c1_state}"
    assert c1_state.get("sufficiency_reason") == "inline_sufficient", "Reason should be inline_sufficient"

    # M109 Debug: Ensure verify was called
    mock_agent.verify_inline_source_relevance.assert_called()
    
    # 3. Final sources should contain our source with SUPPORT stance
    final_sources = result_state.final_sources
    assert len(final_sources) >= 1, "Final sources should not be empty"
    
    found_src = None
    for s in final_sources:
        if s.get("url") == inline_src["url"]:
            found_src = s
            break
            
    assert found_src is not None, "Inline source missing from final output"
    
    # This assertion verifies the fix (auto-stance)
    assert found_src.get("stance") == "SUPPORT", f"Stance should be SUPPORT, got {found_src.get('stance')}"
