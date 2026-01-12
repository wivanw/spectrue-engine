# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.pipeline.steps.audit_claims import AuditClaimsStep
from spectrue_core.pipeline.steps.audit_evidence import AuditEvidenceStep
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext
from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    ContextExcerpt,
    ContextMeta,
    EvidenceItemFrame,
    EvidenceStats,
)
from spectrue_core.schema.rgba_audit import RGBAStatus


class DummyLLM:
    async def call_structured(self, *args, **kwargs):
        return {"bad": "payload"}


def _build_frame() -> ClaimFrame:
    return ClaimFrame(
        claim_id="c1",
        claim_text="Example claim",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="Example claim", span_start=0, span_end=12),
        context_meta=ContextMeta(document_id="doc1"),
        evidence_items=(
            EvidenceItemFrame(
                evidence_id="e1",
                claim_id="c1",
                url="https://example.com",
                source_id="src_example",
                title="Example",
                snippet="Example snippet",
                source_tier="B",
                source_type="news",
                stance="SUPPORT",
            ),
        ),
        evidence_stats=EvidenceStats(total_sources=1, support_sources=1),
    )


@pytest.mark.asyncio
async def test_claim_audit_invalid_output_sets_pipeline_error():
    ctx = PipelineContext(mode=DEEP_MODE, claims=[{"id": "c1", "text": "Example claim"}])
    ctx = ctx.set_extra("deep_claim_ctx", DeepClaimContext(claim_frames=[_build_frame()]))

    result_ctx = await AuditClaimsStep(llm_client=DummyLLM()).run(ctx)
    errors = result_ctx.get_extra("audit_errors") or {}
    claim_errors = errors.get("claim_audit", {})
    assert claim_errors["c1"]["status"] == RGBAStatus.PIPELINE_ERROR


@pytest.mark.asyncio
async def test_evidence_audit_invalid_output_sets_pipeline_error():
    ctx = PipelineContext(mode=DEEP_MODE, claims=[{"id": "c1", "text": "Example claim"}])
    ctx = ctx.set_extra("deep_claim_ctx", DeepClaimContext(claim_frames=[_build_frame()]))

    result_ctx = await AuditEvidenceStep(llm_client=DummyLLM()).run(ctx)
    errors = result_ctx.get_extra("audit_errors") or {}
    evidence_errors = errors.get("evidence_audit", {})
    assert evidence_errors["e1"]["status"] == RGBAStatus.PIPELINE_ERROR
