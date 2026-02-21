# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

import pytest
from unittest.mock import patch

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_V2_MODE
from spectrue_core.pipeline.steps.retrieval.build_cluster_queries import BuildClusterQueriesStep

@pytest.mark.asyncio
async def test_build_cluster_queries_prioritizes_claim_search_queries():
    claims = [
        {"id": "c1", "text": "Claim 1", "search_queries": ["preplanned query 1"]},
        {"id": "c2", "text": "Claim 2", "search_queries": ["preplanned query 2", "another one"]},
    ]
    ctx = PipelineContext(mode=DEEP_V2_MODE, claims=claims)
    ctx = ctx.set_extra("cluster_representatives", {"cluster_1": claims})
    ctx = ctx.set_extra("cluster_claims", {"cluster_1": claims})

    step = BuildClusterQueriesStep()
    new_ctx = await step.run(ctx)
    
    plans = new_ctx.get_extra("cluster_search_plans")
    assert len(plans) == 1
    
    # We expect the preplanned queries to be prioritized
    cluster_queries = plans[0]["search_queries"]
    assert "preplanned query 1" in cluster_queries
    assert "preplanned query 2" in cluster_queries
    assert plans[0].get("query_origin") == "planned"


@pytest.mark.asyncio
async def test_build_cluster_queries_tags_query_origin_and_fallback():
    claims = [
        {"id": "c1", "text": "Claim 1 without queries"},
    ]
    ctx = PipelineContext(mode=DEEP_V2_MODE, claims=claims)
    ctx = ctx.set_extra("cluster_representatives", {"cluster_1": claims})
    ctx = ctx.set_extra("cluster_claims", {"cluster_1": claims})

    step = BuildClusterQueriesStep()
    
    # Force empty cegs
    with patch("spectrue_core.pipeline.steps.retrieval.build_cluster_queries.build_doc_query_plan", return_value=[]):
        new_ctx = await step.run(ctx)
        plans = new_ctx.get_extra("cluster_search_plans")
        
        # It should fall back, use diverse queries, and tag it
        assert plans[0].get("query_origin") == "fallback"
        assert plans[0].get("fallback_reason") in ("no_cegs_queries", "no_queries", "cegs_empty")

@pytest.mark.asyncio
async def test_build_cluster_queries_tags_planned_origin():
    claims = [
        {"id": "c1", "text": "Claim 1 with queries", "search_queries": ["planned"]}
    ]
    ctx = PipelineContext(mode=DEEP_V2_MODE, claims=claims)
    ctx = ctx.set_extra("cluster_representatives", {"cluster_1": claims})
    ctx = ctx.set_extra("cluster_claims", {"cluster_1": claims})

    step = BuildClusterQueriesStep()
    
    new_ctx = await step.run(ctx)
    plans = new_ctx.get_extra("cluster_search_plans")
    
    assert plans[0].get("query_origin") == "planned"
    assert not plans[0].get("fallback_reason")
