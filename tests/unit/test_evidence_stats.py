import pytest
from unittest.mock import MagicMock
from spectrue_core.pipeline.steps.evidence_stats import EvidenceStatsStep
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import AnalysisMode

@pytest.mark.asyncio
async def test_evidence_stats_step_calculation():
    # Setup context
    mode = MagicMock()
    mode.api_analysis_mode = AnalysisMode.DEEP_V2
    
    claims = [{"id": "c1", "text": "claim 1"}]
    sources = [
        {
            "claim_id": "c1",
            "url": "http://domain1.com/p1",
            "domain": "domain1.com",
            "quote": "some quote",
            "covers": ["entity", "time"]
        },
        {
            "claim_id": "c1",
            "url": "http://domain2.com/p2",
            "domain": "domain2.com",
            "quote_span": "another anchor",
            "provenance": "transferred"
        }
    ]
    
    ctx = PipelineContext(mode=mode, claims=claims, sources=sources)
    
    step = EvidenceStatsStep()
    result_ctx = await step.run(ctx)
    
    stats_by_claim = result_ctx.get_extra("evidence_stats_by_claim")
    assert "c1" in stats_by_claim
    stats = stats_by_claim["c1"]
    
    assert stats["sources_observed"] == 2
    assert stats["unique_domains"] == 2
    assert stats["direct_anchors"] == 2
    assert stats["covered_slots"] == 2
    assert stats["transferred"] == 1
    assert 0 <= stats["A_deterministic"] <= 1
    assert stats["A_deterministic"] > 0.5 # Should be decent with 2 sources and anchors
