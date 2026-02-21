# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext, AssembleDeepResultStep
from spectrue_core.schema.claim_frame import ClaimFrame, ContextExcerpt, ContextMeta, EvidenceStats


@pytest.mark.asyncio
async def test_deep_judge_error_returns_null_rgba():
    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Example claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="Example claim", span_start=0, span_end=12),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_stats=EvidenceStats(total_sources=0),
    )

    deep_ctx = DeepClaimContext(
        claim_frames=[frame],
        errors={"c1": {"error_type": "llm_failed", "message": "LLM error"}},
    )

    ctx = PipelineContext(mode=DEEP_MODE, claims=[{"id": "c1", "text": "Example claim"}])
    ctx = ctx.set_extra("deep_claim_ctx", deep_ctx)

    from unittest.mock import patch
    from spectrue_core.utils.trace import Trace

    with patch.object(Trace, "event") as mock_trace:
        result_ctx = await AssembleDeepResultStep().run(ctx)
        
        # Verify no decision_impact events were emitted for an errored claim
        impact_calls = [c for c in mock_trace.call_args_list if c[0][0] == "decision_impact"]
        assert len(impact_calls) == 0, "No decision_impact should be logged on judge error/fallback failure"

    final_result = result_ctx.get_extra("final_result")
    claim_result = final_result["deep_analysis"]["claim_results"][0]

    assert claim_result["status"] == "error"
    assert claim_result["rgba"] is None
    assert claim_result["explanation"] is None
    assert claim_result["error"]["error_type"] == "llm_failed"
    assert "deep_analysis" in final_result
    assert "rgba" not in final_result
