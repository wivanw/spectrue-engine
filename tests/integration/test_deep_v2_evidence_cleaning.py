import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_V2_MODE
from spectrue_core.pipeline.steps.deep_claim import SummarizeEvidenceStep, JudgeClaimsStep, DeepClaimContext
from spectrue_core.schema.claim_frame import ClaimFrame, EvidenceItemFrame, ContextExcerpt, ContextMeta

@pytest.mark.asyncio
async def test_deep_v2_evidence_cleaning_penalizes_boilerplate():
    llm_client = MagicMock()
    
    async def mock_call(*args, **kwargs):
        return {
            "supporting_evidence": [],
            "refuting_evidence": [],
            "contextual_evidence": [{"evidence_id": "e1", "reason": "context"}],
            "evidence_gaps": [],
            "conflicts_present": False
        }
    llm_client.call_structured = AsyncMock(side_effect=mock_call)
    
    async def mock_judge_call(*args, **kwargs):
        return {
            "claim_id": "c1",
            "rgba": {"R": 0.0, "G": 0.8, "B": 1.0, "A": 1.0},
            "confidence": 0.9,
            "verdict": "Supported",
            "explanation": "Test",
            "sources_used": ["http://test.com"],
            "missing_evidence": []
        }
    llm_client.call_json = AsyncMock(side_effect=mock_judge_call)

    # All evidence is boilerplate
    evidence = [
        EvidenceItemFrame(
            evidence_id="e1", 
            claim_id="c1", 
            url="http://test.com", 
            source_id="test",
            snippet="Share this article! Subscribe to our newsletter."
        )
    ]
    
    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Test claim",
        claim_language="en",
        context_excerpt=ContextExcerpt("Test claim in context"),
        context_meta=ContextMeta("doc1"),
        evidence_items=tuple(evidence)
    )
    
    step1 = SummarizeEvidenceStep(llm_client)
    ctx = PipelineContext(mode=DEEP_V2_MODE)
    deep_ctx = DeepClaimContext(claim_frames=[frame])
    ctx = ctx.set_extra("deep_claim_ctx", deep_ctx)
    
    ctx = await step1.run(ctx)
    deep_ctx = ctx.extras["deep_claim_ctx"]
    
    cleaned_frame = deep_ctx.claim_frames[0]
    ev = cleaned_frame.evidence_items[0]
    assert ev.cleanliness is not None
    assert ev.cleanliness.is_boilerplate is True
    
    step2 = JudgeClaimsStep(llm_client)
    ctx = await step2.run(ctx)
    deep_ctx = ctx.extras["deep_claim_ctx"]
    
    out = deep_ctx.judge_outputs["c1"]
    # Base confidence 0.9 - 0.5 penalty = 0.4
    assert out.confidence <= 0.4
