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

from spectrue_core.schema.rgba_audit import RGBAStatus
from spectrue_core.verification.scoring.rgba_audit.aggregation import aggregate_rgba_audit
from tests.fixtures.rgba_audit_fixtures import (
    make_claim_audit,
    make_evidence_audit,
    make_source_metadata,
)


def test_no_evidence_sets_insufficient_evidence_status():
    result = aggregate_rgba_audit(
        claim_audits=[make_claim_audit()],
        evidence_audits=[],
        sources=[],
        trace_context={"events": ["audit_claims.complete"]},
    )

    assert result.G.status == RGBAStatus.INSUFFICIENT_EVIDENCE
    assert result.G.value is None
    assert result.B.status == RGBAStatus.INSUFFICIENT_EVIDENCE


def test_conflicting_evidence_sets_conflict_status_and_trace():
    result = aggregate_rgba_audit(
        claim_audits=[make_claim_audit()],
        evidence_audits=[
            make_evidence_audit(
                evidence_id="e1",
                stance="support",
                directness="direct",
                specificity="high",
                quote_integrity="ok",
                extraction_confidence=0.95,
                audit_confidence=0.9,
                source_id="s1",
            ),
            make_evidence_audit(
                evidence_id="e2",
                stance="refute",
                directness="direct",
                specificity="high",
                quote_integrity="ok",
                extraction_confidence=0.95,
                audit_confidence=0.9,
                source_id="s2",
            ),
        ],
        sources=[
            make_source_metadata(
                source_id="s1",
                tier="A",
                title="Report A",
                snippet="Source A confirms the claim.",
            ),
            make_source_metadata(
                source_id="s2",
                tier="A",
                title="Report B",
                snippet="Source B contradicts the claim.",
            ),
        ],
        trace_context={"events": ["audit_evidence.complete"]},
    )

    assert result.G.status == RGBAStatus.CONFLICTING_EVIDENCE
    assert result.G.value is None
    assert "support_mass" in result.G.trace
    assert "refute_mass" in result.G.trace


def test_pipeline_error_sets_pipeline_error_status():
    result = aggregate_rgba_audit(
        claim_audits=[make_claim_audit()],
        evidence_audits=[make_evidence_audit()],
        sources=[make_source_metadata()],
        trace_context={"events": ["audit_evidence.complete"]},
        audit_errors={"claim_audit": {"c1": {"status": RGBAStatus.PIPELINE_ERROR}}},
    )

    assert result.G.status == RGBAStatus.PIPELINE_ERROR
    assert result.G.value is None
