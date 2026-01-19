import pytest
from spectrue_core.pipeline.steps.deep_claim import AssembleDeepResultStep, DeepClaimContext
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.schema.claim_frame import ClaimFrame, JudgeOutput, RGBAScore, ContextExcerpt, ContextMeta

def _create_mock_frame(cid: str, text: str) -> ClaimFrame:
    return ClaimFrame(
        claim_id=cid,
        claim_text=text,
        claim_language="en",
        context_excerpt=ContextExcerpt(text="context"),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_items=()
    )

def _create_mock_judge_output(cid: str) -> JudgeOutput:
    return JudgeOutput(
        claim_id=cid,
        rgba=RGBAScore(r=0.1, g=0.9, b=0.9, a=0.9),
        confidence=0.9,
        verdict="Supported",
        explanation="Test explanation",
        sources_used=()
    )

@pytest.mark.asyncio
async def test_inv_001_claims_are_atomic():
    """
    INV-001: Claims are Atomic and Non-Merging.
    Verify that multiple distinct claims in the input are preserved as distinct
    outputs in the final result, mapped 1:1 by claim_id.
    """
    frames = [
        _create_mock_frame("c1", "Claim 1"),
        _create_mock_frame("c2", "Claim 2")
    ]
    
    # Mock judge outputs for them
    judge_outputs = {
        "c1": _create_mock_judge_output("c1"),
        "c2": _create_mock_judge_output("c2")
    }
    
    deep_ctx = DeepClaimContext(
        claim_frames=frames,
        judge_outputs=judge_outputs
    )
    
    ctx = PipelineContext(
        mode=DEEP_MODE,
        extras={"deep_claim_ctx": deep_ctx}
    )
    
    step = AssembleDeepResultStep()
    result_ctx = await step.run(ctx)
    result_data = result_ctx.get_extra("deep_analysis_result")
    
    # Assertions
    assert result_data is not None
    assert "claim_results" in result_data
    results = result_data["claim_results"]
    
    # Must have 2 distinct results
    assert len(results) == 2
    
    ids = sorted([r["claim_id"] for r in results])
    assert ids == ["c1", "c2"]
    
    # Verify no merging logic happened (e.g. unique texts)
    texts = sorted([r["claim_text"] for r in results])
    assert texts == ["Claim 1", "Claim 2"]

@pytest.mark.asyncio
async def test_inv_002_no_aggregation_without_claims():
    """
    INV-002: No Aggregation Without Per-Claim Outputs.
    Verify that the final payload includes the list of claims/results
    alongside any aggregate or verdict info.
    """
    frame = _create_mock_frame("c1", "Claim 1")
    judge_output = _create_mock_judge_output("c1")
    
    deep_ctx = DeepClaimContext(
        claim_frames=[frame],
        judge_outputs={"c1": judge_output}
    )
    
    ctx = PipelineContext(
        mode=DEEP_MODE,
        extras={"deep_claim_ctx": deep_ctx}
    )
    
    step = AssembleDeepResultStep()
    result_ctx = await step.run(ctx)
    
    # Check Verdict object
    verdict = result_ctx.verdict
    assert verdict is not None
    
    # Check Deep Analysis payload
    deep_result = result_ctx.get_extra("deep_analysis_result")
    assert deep_result is not None
    
    # The Invariant: "claims" or "claim_results" MUST be present
    # In Deep Mode v5, we return "claim_results" inside "deep_analysis"
    assert "claim_results" in deep_result
    assert len(deep_result["claim_results"]) > 0
    
    # And specifically inside the top-level Verdict if used
    if "claim_verdicts" in verdict:
        assert len(verdict["claim_verdicts"]) > 0
