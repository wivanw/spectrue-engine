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
"""RGBA audit schema contracts and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from spectrue_core.schema.serialization import SchemaModel


class RGBAStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNVERIFIABLE_BY_NATURE = "UNVERIFIABLE_BY_NATURE"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


RGBA_STATUS_LEGACY_CODES: dict[RGBAStatus, int] = {
    RGBAStatus.OK: 0,
    RGBAStatus.INSUFFICIENT_EVIDENCE: -1,
    RGBAStatus.CONFLICTING_EVIDENCE: -2,
    RGBAStatus.UNVERIFIABLE_BY_NATURE: -3,
    RGBAStatus.PIPELINE_ERROR: -4,
    RGBAStatus.OUT_OF_SCOPE: -5,
}

LEGACY_CODE_TO_STATUS: dict[int, RGBAStatus] = {
    value: key for key, value in RGBA_STATUS_LEGACY_CODES.items()
}


def rgba_status_to_legacy_code(status: RGBAStatus) -> int:
    return RGBA_STATUS_LEGACY_CODES[status]


def legacy_code_to_rgba_status(code: int) -> RGBAStatus | None:
    return LEGACY_CODE_TO_STATUS.get(int(code))


class RGBAMetric(SchemaModel):
    status: RGBAStatus
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: dict[str, Any] | None = None
    reasons: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_status_constraints(self) -> "RGBAMetric":
        if self.status != RGBAStatus.OK:
            if self.value is not None:
                raise ValueError("value must be null when status is not OK")
            if self.confidence is not None or self.uncertainty is not None:
                raise ValueError("confidence/uncertainty only allowed when status is OK")
        return self

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "status_code": rgba_status_to_legacy_code(self.status),
            "value": self.value,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "reasons": list(self.reasons),
            "trace": dict(self.trace),
        }


class RGBAResult(SchemaModel):
    R: RGBAMetric
    G: RGBAMetric
    B: RGBAMetric
    A: RGBAMetric
    global_reasons: list[str] = Field(default_factory=list)
    summary_trace: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "R": self.R.to_payload(),
            "G": self.G.to_payload(),
            "B": self.B.to_payload(),
            "A": self.A.to_payload(),
            "global_reasons": list(self.global_reasons),
            "summary_trace": dict(self.summary_trace),
        }


class ClaimAudit(SchemaModel):
    claim_id: str
    predicate_type: Literal[
        "event",
        "measurement",
        "quote",
        "policy",
        "ranking",
        "causal",
        "other",
    ]
    truth_conditions: list[str]
    expected_evidence_types: list[str]
    failure_modes: list[str]
    assertion_strength: Literal["weak", "medium", "strong"]
    risk_facets: list[str]
    honesty_facets: list[str]
    what_would_change_mind: list[str]
    audit_confidence: float = Field(ge=0.0, le=1.0)


class EvidenceAudit(SchemaModel):
    claim_id: str
    evidence_id: str
    source_id: str
    stance: Literal["support", "refute", "unclear", "unrelated"]
    directness: Literal["direct", "indirect", "tangential"]
    specificity: Literal["high", "medium", "low"]
    quote_integrity: Literal["ok", "partial", "out_of_context", "not_applicable"]
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    novelty_vs_copy: Literal["original", "syndicated", "unknown"]
    dependency_hints: list[str]
    audit_confidence: float = Field(ge=0.0, le=1.0)


class SourceCluster(SchemaModel):
    cluster_id: str
    source_ids: list[str]
    representative_source_id: str
    size: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_size(self) -> "SourceCluster":
        if self.size != len(self.source_ids):
            raise ValueError("size must match number of source_ids")
        return self
