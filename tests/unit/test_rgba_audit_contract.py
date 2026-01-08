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

from spectrue_core.schema.rgba_audit import (
    RGBAStatus,
    RGBAMetric,
    RGBAResult,
    legacy_code_to_rgba_status,
    rgba_status_to_legacy_code,
)


def test_rgba_status_legacy_mapping_roundtrip():
    assert rgba_status_to_legacy_code(RGBAStatus.OK) == 0
    assert rgba_status_to_legacy_code(RGBAStatus.CONFLICTING_EVIDENCE) == -2
    assert legacy_code_to_rgba_status(-2) == RGBAStatus.CONFLICTING_EVIDENCE
    assert legacy_code_to_rgba_status(0) == RGBAStatus.OK


def test_rgba_metric_requires_null_value_for_non_ok():
    with pytest.raises(ValueError):
        RGBAMetric(
            status=RGBAStatus.INSUFFICIENT_EVIDENCE,
            value=0.4,
            reasons=[],
            trace={},
        )


def test_rgba_metric_confidence_only_when_ok():
    with pytest.raises(ValueError):
        RGBAMetric(
            status=RGBAStatus.PIPELINE_ERROR,
            value=None,
            confidence=0.5,
            reasons=[],
            trace={},
        )

    metric = RGBAMetric(
        status=RGBAStatus.OK,
        value=0.6,
        confidence=0.7,
        reasons=["enough_evidence"],
        trace={"support": 2},
    )
    payload = metric.to_payload()
    assert payload["status"] == "OK"
    assert payload["status_code"] == 0
    assert payload["value"] == 0.6


def test_rgba_result_payload_serialization():
    ok_metric = RGBAMetric(
        status=RGBAStatus.OK,
        value=0.8,
        confidence=0.9,
        reasons=["supporting_evidence"],
        trace={"support": 3},
    )
    no_evidence = RGBAMetric(
        status=RGBAStatus.INSUFFICIENT_EVIDENCE,
        value=None,
        reasons=["no_sources"],
        trace={"support": 0, "refute": 0},
    )

    result = RGBAResult(
        R=ok_metric,
        G=no_evidence,
        B=no_evidence,
        A=ok_metric,
        global_reasons=["audit_pass"],
        summary_trace={"clusters": 1},
    )

    payload = result.to_payload()
    assert payload["G"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["G"]["status_code"] == -1
    assert payload["G"]["value"] is None
    assert payload["global_reasons"] == ["audit_pass"]
