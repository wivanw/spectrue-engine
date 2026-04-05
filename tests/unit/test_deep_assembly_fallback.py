import pytest
from unittest.mock import MagicMock
from spectrue_core.pipeline.steps.deep_claim import AssembleDeepResultStep, DeepClaimContext
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.schema.claim_frame import ClaimFrame, JudgeOutput

@pytest.mark.asyncio
async def test_assemble_deep_result_no_fallback_A():
    # Setup context
    mode = MagicMock()
    mode.api_analysis_mode = AnalysisMode.DEEP_V2
    
    # Claim with A = -1.0 from judge
    frame = MagicMock(spec=ClaimFrame)
    frame.claim_id = "c1"
    frame.claim_text = "test claim"
    frame.evidence_items = []
    frame.evidence_stats = MagicMock()
    frame.confirmation_counts = MagicMock()
    
    judge_output = MagicMock(spec=JudgeOutput)
    judge_output.rgba = MagicMock()
    judge_output.rgba.r = 0.1
    judge_output.rgba.g = 0.8
    judge_output.rgba.b = 0.9
    judge_output.rgba.a = -1.0 # Invalid A
    judge_output.explanation = "reason"
    judge_output.sources_used = []
    judge_output.verdict = "verified"
    judge_output.confidence = 0.9
    
    deep_ctx = DeepClaimContext()
    deep_ctx.claim_frames = [frame]
    deep_ctx.judge_outputs = {"c1": judge_output}
    
    ctx = PipelineContext(mode=mode)
    ctx = ctx.set_extra("deep_claim_ctx", deep_ctx)
    
    step = AssembleDeepResultStep()
    result_ctx = await step.run(ctx)
    
    # Verify no fallback happened (A stays -1.0)
    final_result = result_ctx.get_extra("final_result")
    claim_res = final_result["deep_analysis"]["claim_results"][0]
    assert claim_res["rgba"][3] == -1.0
