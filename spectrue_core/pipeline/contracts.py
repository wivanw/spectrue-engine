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

"""Pipeline step contracts for stable, typed handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

INPUT_DOC_KEY = "input_doc"
CLAIMS_KEY = "claims_contract"
EVIDENCE_INDEX_KEY = "evidence_index"
JUDGMENTS_KEY = "judgments"


@dataclass(frozen=True, slots=True)
class InputDoc:
    """Normalized representation of the user input."""

    input_type: Literal["url", "text"]
    raw_input: str
    prepared_text: str
    lang: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "raw_input": self.raw_input,
            "prepared_text": self.prepared_text,
            "lang": self.lang,
        }


@dataclass(frozen=True, slots=True)
class ClaimItem:
    """Single extracted claim with text span metadata."""

    id: str
    text: str
    span_start: int | None = None
    span_end: int | None = None
    lang: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "lang": self.lang,
        }


@dataclass(frozen=True, slots=True)
class Claims:
    """Collection of claims plus optional anchor ID."""

    claims: tuple[ClaimItem, ...] = field(default_factory=tuple)
    anchor_claim_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_payload() for claim in self.claims],
            "anchor_claim_id": self.anchor_claim_id,
        }


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Single evidence item for contract-level handoff."""

    url: str
    title: str | None = None
    snippet: str | None = None
    quote: str | None = None
    provider_score: float | None = None
    sim: float | None = None
    stance: str | None = None
    relevance: float | None = None
    tier: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "quote": self.quote,
            "provider_score": self.provider_score,
            "sim": self.sim,
            "stance": self.stance,
            "relevance": self.relevance,
            "tier": self.tier,
        }


@dataclass(frozen=True, slots=True)
class EvidencePackContract:
    """Evidence bundle for a single claim or global scope."""

    items: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    stats: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "items": [item.to_payload() for item in self.items],
            "stats": dict(self.stats),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class EvidenceIndex:
    """Evidence packs for standard (global) or deep (per-claim) modes."""

    by_claim_id: dict[str, EvidencePackContract] = field(default_factory=dict)
    global_pack: EvidencePackContract | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "by_claim_id": {
                claim_id: pack.to_payload()
                for claim_id, pack in self.by_claim_id.items()
            },
            "global": self.global_pack.to_payload() if self.global_pack else None,
        }


@dataclass(frozen=True, slots=True)
class Judgments:
    """Judgment outputs for standard or deep modes."""

    standard: dict[str, Any] | None = None
    deep: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        return {
            "standard": dict(self.standard) if self.standard else None,
            "deep": [dict(item) for item in self.deep],
        }
