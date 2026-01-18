import pytest
from spectrue_core.pipeline.steps.deep_claim import JudgeClaimsStep, DeepClaimContext
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.schema.claim_frame import ClaimFrame, ContextExcerpt, ContextMeta
from spectrue_core.agents.llm_client import LLMFailureKind, LLMCallError

@pytest.mark.asyncio
async def test_inv_040_no_fallback_0_5(mock_llm_client):
    """
    INV-040: No "Fallback 0.5" -> Null + Status.
    Verify that when LLM fails, we get status='error' and rgba=None, not [0.5...].
    """
    # Simulate LLM failure
    mock_llm_client.call_structured.side_effect = LLMCallError(
        message="Simulated provider failure",
        kind=LLMFailureKind.PROVIDER_ERROR
    )
    
    step = JudgeClaimsStep(llm_client=mock_llm_client)
    
    # Correct ClaimFrame instantiation
    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Test claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="snippet surrounding claim"),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_items=()
    )
    
    deep_ctx = DeepClaimContext(claim_frames=[frame])
    
    ctx = PipelineContext(
        mode=DEEP_MODE,
        extras={"deep_claim_ctx": deep_ctx}
    )
    
    # Run the step
    result_ctx = await step.run(ctx)
    result_deep_ctx = result_ctx.get_extra("deep_claim_ctx")
    
    # Check errors
    assert "c1" in result_deep_ctx.errors
    error = result_deep_ctx.errors["c1"]
    assert error["error_type"] == "llm_failed"
    
    # Check outputs - should NOT be present for c1 if error occurred
    assert "c1" not in result_deep_ctx.judge_outputs

@pytest.mark.asyncio
async def test_inv_041_schema_enforcement(mock_llm_client):
    """
    INV-041: Schema Validation is Mandatory.
    """
    # Simulate LLM returning garbage
    mock_llm_client.call_structured.side_effect = LLMCallError(
        message="Schema validation failed",
        kind=LLMFailureKind.INVALID_JSON 
    )

    step = JudgeClaimsStep(llm_client=mock_llm_client)
    
    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Test claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="snippet surrounding claim"),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_items=()
    )
    
    deep_ctx = DeepClaimContext(claim_frames=[frame])
    
    ctx = PipelineContext(
        mode=DEEP_MODE,
        extras={"deep_claim_ctx": deep_ctx}
    )
    
    result_ctx = await step.run(ctx)
    result_deep_ctx = result_ctx.get_extra("deep_claim_ctx")
    
    assert "c1" in result_deep_ctx.errors
    assert "c1" not in result_deep_ctx.judge_outputs

@pytest.mark.asyncio
async def test_inv_042_score_is_deterministic(mock_llm_client):
    """
    INV-042: LLM Signals Only.
    Verify that the system propagates exact LLM scores without "fudging" or inventing numbers.
    """
    from spectrue_core.schema.claim_frame import RGBAScore
    
    # Mock specific return values
    # Note: We can't easily mock the internal parsing if we don't mock the LLM response JSON directly.
    # So we'll mock call_json to return a specific dict.
    
    target_rgba = {"R": 0.123, "G": 0.456, "B": 0.789, "A": 0.999}
    
    # Mock specific return values
    # ClaimJudgeSkill uses 'call_structured', not 'call_json'
    mock_llm_client.call_structured.return_value = {
        "claim_id": "c1",
        "rgba": target_rgba,
        "confidence": 0.888,
        "verdict": "Likely",
        "explanation": "Exp",
        "sources_used": []
    }
    
    step = JudgeClaimsStep(llm_client=mock_llm_client)
    
    from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext
    from spectrue_core.schema.claim_frame import ClaimFrame, ContextExcerpt, ContextMeta
    
    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Test",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="c"),
        context_meta=ContextMeta(document_id="d"),
        evidence_items=()
    )
    
    deep_ctx = DeepClaimContext(claim_frames=[frame])
    ctx = PipelineContext(mode=DEEP_MODE, extras={"deep_claim_ctx": deep_ctx})
    
    # Run
    result_ctx = await step.run(ctx)
    result_deep_ctx = result_ctx.get_extra("deep_claim_ctx")
    
    output = result_deep_ctx.judge_outputs.get("c1")
    assert output is not None
    
    # Check exact propagation
    # Note: parsing might clamp or validate, but if valid range, should be close.
    # INV-042 allows valid normalization (clamping), but strictly forbids "invention" (magic numbers).
    
    assert abs(output.rgba.r - 0.123) < 0.0001
    assert abs(output.rgba.g - 0.456) < 0.0001
    assert abs(output.rgba.b - 0.789) < 0.0001
    assert abs(output.rgba.a - 0.999) < 0.0001
    assert output.confidence == 0.888
