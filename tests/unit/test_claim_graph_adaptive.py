import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.steps.claim_graph import ClaimGraphStep

from spectrue_core.pipeline.mode import get_mode

@pytest.mark.asyncio
async def test_adaptive_graph_skip():
    mock_builder = MagicMock()
    runtime_config = MagicMock()
    # Cost per claim is 1.0, budget is 10.0
    runtime_config.claim_graph.selection_budget = 10.0
    runtime_config.claim_graph.default_claim_cost = 1.0
    
    step = ClaimGraphStep(mock_builder, runtime_config)
    
    # 5 claims, total cost 5.0 < 10.0 -> should skip graph
    claims = [{"id": f"c{i}"} for i in range(5)]
    ctx = PipelineContext(mode=get_mode("general")).with_update(claims=claims)
    
    ctx = await step.run(ctx)
    assert ctx.extras.get("graph_result") is None
    assert ctx.extras.get("key_claim_ids") == [f"c{i}" for i in range(5)]

@pytest.mark.asyncio
async def test_adaptive_graph_execute():
    mock_result = MagicMock()
    mock_result.key_claim_ids = ["c1", "c2"]
    mock_result.graph_result = "graph_res"
    mock_result.disabled = False
    mock_result.to_trace_dict.return_value = {}
    mock_result.get_ranked_by_id.return_value = {}
    
    mock_builder = AsyncMock()
    # Mock return value
    mock_builder.build = AsyncMock(return_value=mock_result)
    
    runtime_config = MagicMock()
    # Cost per claim is 1.0, budget is 2.0
    runtime_config.claim_graph.selection_budget = 2.0
    runtime_config.claim_graph.default_claim_cost = 1.0
    
    step = ClaimGraphStep(mock_builder, runtime_config)
    
    # 5 claims, total cost 5.0 > 2.0 -> should execute graph
    claims = [{"id": f"c{i}"} for i in range(5)]
    ctx = PipelineContext(mode=get_mode("general")).with_update(claims=claims)
    
    ctx = await step.run(ctx)
    assert ctx.extras.get("graph_result") is not None
    assert set(ctx.extras.get("key_claim_ids")) == {"c1", "c2"}
