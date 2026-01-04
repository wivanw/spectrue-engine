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

    result_ctx = await AssembleDeepResultStep().run(ctx)
    final_result = result_ctx.get_extra("final_result")

    assert final_result["deep_analysis"]["claim_results"][0]["status"] == "error"
    assert final_result["deep_analysis"]["claim_results"][0]["rgba"] is None
