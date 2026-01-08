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

from spectrue_core.pipeline.contracts import RGBA_AUDIT_KEY
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.pipeline.steps.aggregate_rgba_audit import AggregateRGBAAuditStep
from spectrue_core.schema.rgba_audit import RGBAStatus
from tests.fixtures.rgba_audit_fixtures import (
    make_claim_audit,
    make_evidence_audit,
    make_source_metadata,
)


@pytest.mark.asyncio
async def test_rgba_audit_pipeline_conflict():
    ctx = PipelineContext(mode=DEEP_MODE)
    ctx = ctx.set_extra("claim_audits", [make_claim_audit()])
    ctx = ctx.set_extra(
        "evidence_audits",
        [
            make_evidence_audit(evidence_id="e1", stance="support", source_id="s1"),
            make_evidence_audit(evidence_id="e2", stance="refute", source_id="s2"),
        ],
    )
    ctx = ctx.set_extra(
        "audit_sources",
        [
            make_source_metadata(
                source_id="s1",
                title="Report A",
                snippet="Source A confirms the claim.",
            ),
            make_source_metadata(
                source_id="s2",
                title="Report B",
                snippet="Source B contradicts the claim.",
            ),
        ],
    )
    ctx = ctx.set_extra("audit_trace_context", {"events": ["audit_evidence.complete"]})

    result_ctx = await AggregateRGBAAuditStep().run(ctx)
    rgba_result = result_ctx.get_extra(RGBA_AUDIT_KEY)

    assert rgba_result.G.status == RGBAStatus.CONFLICTING_EVIDENCE
    assert rgba_result.G.value is None


@pytest.mark.asyncio
async def test_rgba_audit_pipeline_no_evidence():
    ctx = PipelineContext(mode=DEEP_MODE)
    ctx = ctx.set_extra("claim_audits", [make_claim_audit()])
    ctx = ctx.set_extra("evidence_audits", [])
    ctx = ctx.set_extra("audit_trace_context", {"events": ["audit_claims.complete"]})

    result_ctx = await AggregateRGBAAuditStep().run(ctx)
    rgba_result = result_ctx.get_extra(RGBA_AUDIT_KEY)

    assert rgba_result.G.status == RGBAStatus.INSUFFICIENT_EVIDENCE
    assert rgba_result.G.value is None


@pytest.mark.asyncio
async def test_rgba_audit_pipeline_error():
    ctx = PipelineContext(mode=DEEP_MODE)
    ctx = ctx.set_extra("claim_audits", [make_claim_audit()])
    ctx = ctx.set_extra("evidence_audits", [make_evidence_audit()])
    ctx = ctx.set_extra("audit_errors", {"claim_audit": {"c1": {"status": RGBAStatus.PIPELINE_ERROR}}})
    ctx = ctx.set_extra("audit_trace_context", {"events": ["audit_claims.complete"]})

    result_ctx = await AggregateRGBAAuditStep().run(ctx)
    rgba_result = result_ctx.get_extra(RGBA_AUDIT_KEY)

    assert rgba_result.G.status == RGBAStatus.PIPELINE_ERROR
    assert rgba_result.G.value is None
