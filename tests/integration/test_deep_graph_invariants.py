# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from unittest.mock import MagicMock

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext, AssembleDeepResultStep
from spectrue_core.pipeline.steps.metering_setup import METERING_SETUP_STEP_NAME
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    ContextExcerpt,
    ContextMeta,
    EvidenceStats,
    JudgeOutput,
    RGBAScore,
)


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.features = MagicMock()
    return config


@pytest.fixture
def pipeline_factory():
    return PipelineFactory(search_mgr=MagicMock(), agent=MagicMock())


def test_deep_graph_invariants(mock_config, pipeline_factory):
    pipeline = pipeline_factory.build("deep", config=mock_config)
    names = [node.name for node in pipeline.nodes]

    assert METERING_SETUP_STEP_NAME in names
    assert "judge_claims" in names
    assert "judge_standard" not in names


@pytest.mark.asyncio
async def test_deep_result_contract_fields():
    ok_frame = ClaimFrame(
        claim_id="c1",
        claim_text="Example claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="Example claim", span_start=0, span_end=12),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_stats=EvidenceStats(total_sources=1),
    )
    error_frame = ClaimFrame(
        claim_id="c2",
        claim_text="Another claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="Another claim", span_start=0, span_end=13),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_stats=EvidenceStats(total_sources=0),
    )

    deep_ctx = DeepClaimContext(
        claim_frames=[ok_frame, error_frame],
        judge_outputs={
            "c1": JudgeOutput(
                claim_id="c1",
                rgba=RGBAScore(r=0.1, g=0.2, b=0.3, a=0.4),
                confidence=0.5,
                verdict="Supported",
                explanation="Example explanation",
                sources_used=(),
                missing_evidence=(),
            ),
        },
        errors={"c2": {"error_type": "llm_failed", "message": "LLM error"}},
    )

    ctx = PipelineContext(mode=DEEP_MODE, claims=[{"id": "c1"}, {"id": "c2"}])
    ctx = ctx.set_extra("deep_claim_ctx", deep_ctx)

    result_ctx = await AssembleDeepResultStep().run(ctx)
    final_result = result_ctx.get_extra("final_result")

    assert final_result["judge_mode"] == "deep"
    assert "deep_analysis" in final_result
    assert len(final_result["deep_analysis"]["claim_results"]) == 2
    assert "rgba" not in final_result
    assert "anchor_claim" not in final_result
