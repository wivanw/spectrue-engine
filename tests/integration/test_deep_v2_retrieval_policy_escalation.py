# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_V2_MODE
from spectrue_core.pipeline.steps.retrieval.cluster_web_search import ClusterWebSearchStep
from spectrue_core.runtime_config import DeepV2Config

@pytest.mark.asyncio
async def test_deep_v2_escalates_to_academic_for_scientific_claims():
    """
    Integration regression (US2): Ensure that scientific claims escalate to 
    academic search when trusted and open searches are insufficient.
    """
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.deep_v2 = DeepV2Config()

    search_mgr = MagicMock()
    
    # We mock search phase to always return empty/insufficient results
    # so we can intercept all 3 tiers of escalation
    call_kwargs_list = []
    
    async def _search_phase(query, **kwargs):
        call_kwargs_list.append(kwargs)
        return None, []

    search_mgr.search_phase = AsyncMock(side_effect=_search_phase)
    search_mgr.fetch_urls_content_batch = AsyncMock(return_value={})

    # Claim with scientific policy mode
    claims = [
        {"id": "c_sci", "text": "A new scientific paper study shows...", "policy_mode": "SCIENTIFIC"},
    ]

    ctx_v2 = PipelineContext(mode=DEEP_V2_MODE, claims=claims, lang="en")
    
    plan = {
        "cluster_id": "cluster_1",
        "search_queries": ["science study"],
        "claims": claims,
    }
    ctx_v2 = ctx_v2.set_extra("cluster_search_plans", [plan])

    ctx_v2 = await ClusterWebSearchStep(config=config, search_mgr=search_mgr).run(ctx_v2)

    assert len(call_kwargs_list) == 3, "Expected 3 search passes: trusted, open, academic"
    
    # Pass 1: Trusted
    assert "include_domains" in call_kwargs_list[0]
    # Pass 2: Open
    assert "exclude_domains" in call_kwargs_list[1]
    # Pass 3: Scientific escalation (Tavily topic must be general|news|finance)
    assert call_kwargs_list[2].get("topic") == "general" or "include_domains" in call_kwargs_list[2]

@pytest.mark.asyncio
async def test_deep_v2_does_not_escalate_to_academic_for_general_claims():
    """
    Integration regression (US2): Ensure that general claims DO NOT escalate to 
    academic search, even if trusted and open searches are insufficient.
    """
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.deep_v2 = DeepV2Config()

    search_mgr = MagicMock()
    call_kwargs_list = []
    
    async def _search_phase(query, **kwargs):
        call_kwargs_list.append(kwargs)
        return None, []

    search_mgr.search_phase = AsyncMock(side_effect=_search_phase)
    search_mgr.fetch_urls_content_batch = AsyncMock(return_value={})

    # Claim with general policy mode
    claims = [
        {"id": "c_gen", "text": "The mayor said he loves pizza...", "policy_mode": "STANDARD"},
    ]

    ctx_v2 = PipelineContext(mode=DEEP_V2_MODE, claims=claims, lang="en")
    plan = {
        "cluster_id": "cluster_2",
        "search_queries": ["mayor pizza"],
        "claims": claims,
    }
    ctx_v2 = ctx_v2.set_extra("cluster_search_plans", [plan])

    ctx_v2 = await ClusterWebSearchStep(config=config, search_mgr=search_mgr).run(ctx_v2)

    assert len(call_kwargs_list) == 2, "Expected 2 search passes: trusted, open. NO academic."
    
    # Pass 1: Trusted
    assert "include_domains" in call_kwargs_list[0]
    # Pass 2: Open
    assert "exclude_domains" in call_kwargs_list[1]
