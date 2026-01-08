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

"""Shared synthetic fixtures for RGBA audit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CLAIM_AUDIT: dict[str, Any] = {
    "claim_id": "c1",
    "predicate_type": "event",
    "truth_conditions": ["The event occurred as described."],
    "expected_evidence_types": ["official_statement", "news_report"],
    "failure_modes": ["no_sources", "conflicting_sources"],
    "assertion_strength": "strong",
    "risk_facets": ["public_safety"],
    "honesty_facets": ["selective_context"],
    "what_would_change_mind": ["Official report contradicting the claim."],
    "audit_confidence": 0.82,
}

DEFAULT_EVIDENCE_AUDIT: dict[str, Any] = {
    "claim_id": "c1",
    "evidence_id": "e1",
    "source_id": "s1",
    "stance": "support",
    "directness": "direct",
    "specificity": "high",
    "quote_integrity": "ok",
    "extraction_confidence": 0.9,
    "novelty_vs_copy": "original",
    "dependency_hints": [],
    "audit_confidence": 0.76,
}

DEFAULT_SOURCE_METADATA: dict[str, Any] = {
    "source_id": "s1",
    "title": "Example Source",
    "snippet": "Example snippet describing the claim.",
    "channel_type": "news",
    "tier": "B",
}


def make_claim_audit(**overrides: Any) -> dict[str, Any]:
    payload = deepcopy(DEFAULT_CLAIM_AUDIT)
    payload.update(overrides)
    return payload


def make_evidence_audit(**overrides: Any) -> dict[str, Any]:
    payload = deepcopy(DEFAULT_EVIDENCE_AUDIT)
    payload.update(overrides)
    return payload


def make_source_metadata(**overrides: Any) -> dict[str, Any]:
    payload = deepcopy(DEFAULT_SOURCE_METADATA)
    payload.update(overrides)
    return payload


def sample_claim_audits() -> list[dict[str, Any]]:
    return [
        make_claim_audit(claim_id="c1"),
        make_claim_audit(claim_id="c2", assertion_strength="medium"),
    ]


def sample_evidence_audits() -> list[dict[str, Any]]:
    return [
        make_evidence_audit(claim_id="c1", evidence_id="e1", source_id="s1"),
        make_evidence_audit(
            claim_id="c1",
            evidence_id="e2",
            source_id="s2",
            stance="refute",
            directness="indirect",
            specificity="medium",
            quote_integrity="partial",
            novelty_vs_copy="syndicated",
            extraction_confidence=0.65,
        ),
    ]


def sample_sources() -> list[dict[str, Any]]:
    return [
        make_source_metadata(source_id="s1", tier="A"),
        make_source_metadata(source_id="s2", tier="C"),
    ]
