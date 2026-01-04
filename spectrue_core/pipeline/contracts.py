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
STANDARD_JUDGMENT_KEY = "standard_judgment"
DEEP_CLAIM_RESULTS_KEY = "deep_claim_results"
FINAL_PAYLOAD_KEY = "final_payload"
SEARCH_PLAN_KEY = "search_plan"
RAW_SEARCH_RESULTS_KEY = "raw_search_results"
RANKED_RESULTS_KEY = "ranked_results"
RETRIEVAL_ITEMS_KEY = "retrieval_items"


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
class SearchPlan:
    """Structured query plan for retrieval."""

    plan_id: str
    mode: Literal["standard", "deep"]
    global_queries: tuple[str, ...] = field(default_factory=tuple)
    per_claim_queries: dict[str, tuple[str, ...]] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "mode": self.mode,
            "global_queries": list(self.global_queries),
            "per_claim_queries": {
                claim_id: list(queries)
                for claim_id, queries in self.per_claim_queries.items()
            },
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class RawSearchResult:
    """Raw provider response tied to a single query."""

    plan_id: str
    query_id: str
    query: str
    claim_id: str | None
    provider_payload: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query_id": self.query_id,
            "query": self.query,
            "claim_id": self.claim_id,
            "provider_payload": list(self.provider_payload),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class RawSearchResults:
    """Collection of raw search responses for a plan."""

    plan_id: str
    results: tuple[RawSearchResult, ...] = field(default_factory=tuple)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "results": [result.to_payload() for result in self.results],
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class RankedResultItem:
    """Ranked search result with blended score."""

    url: str
    provider_score: float
    similarity_score: float
    blended_score: float
    claim_id: str | None = None
    title: str | None = None
    snippet: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "provider_score": self.provider_score,
            "similarity_score": self.similarity_score,
            "blended_score": self.blended_score,
            "claim_id": self.claim_id,
            "title": self.title,
            "snippet": self.snippet,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True, slots=True)
class RankedResults:
    """Ranked search results grouped by scope."""

    plan_id: str
    results_global: tuple[RankedResultItem, ...] = field(default_factory=tuple)
    results_by_claim: dict[str, tuple[RankedResultItem, ...]] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "results_global": [item.to_payload() for item in self.results_global],
            "results_by_claim": {
                claim_id: [item.to_payload() for item in items]
                for claim_id, items in self.results_by_claim.items()
            },
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class RetrievalItem:
    """Normalized retrieval item used for evidence packaging."""

    url: str
    provider_score: float | None = None
    similarity_score: float | None = None
    blended_score: float | None = None
    claim_id: str | None = None
    title: str | None = None
    snippet: str | None = None
    quote: str | None = None
    content_excerpt: str | None = None
    rank: int | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "provider_score": self.provider_score,
            "similarity_score": self.similarity_score,
            "blended_score": self.blended_score,
            "claim_id": self.claim_id,
            "title": self.title,
            "snippet": self.snippet,
            "quote": self.quote,
            "content_excerpt": self.content_excerpt,
            "rank": self.rank,
            "trace": dict(self.trace),
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
    stats: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    missing_claims: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        return {
            "global_pack": self.global_pack.to_payload() if self.global_pack else None,
            "by_claim": {
                claim_id: pack.to_payload()
                for claim_id, pack in self.by_claim_id.items()
            },
            "stats": dict(self.stats),
            "trace": dict(self.trace),
            "missing_claims": list(self.missing_claims),
        }


@dataclass(frozen=True, slots=True)
class StandardJudgment:
    """Standard mode judgment output from article-level LLM judge.

    All scores are post-processed (penalties applied, tier caps enforced).
    """

    verdict: str
    verdict_score: float
    rgba: tuple[float, float, float, float]
    rationale: str
    analysis: str
    details: tuple[str, ...] = field(default_factory=tuple)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_score": self.verdict_score,
            "rgba": list(self.rgba),
            "rationale": self.rationale,
            "analysis": self.analysis,
            "details": list(self.details),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class DeepClaimResult:
    """Per-claim judgment result for deep mode.

    Contract:
    - status="ok": rgba and verdict_score are populated
    - status="error": rgba=None, verdict_score=None, error_reason populated
    - NO fallback 0.5 scores — null on error
    """

    claim_id: str
    status: Literal["ok", "error"]
    verdict: str | None = None
    verdict_score: float | None = None
    rgba: tuple[float, float, float, float] | None = None
    explanation: str | None = None
    sources_used: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    error_reason: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "verdict": self.verdict,
            "verdict_score": self.verdict_score,
            "rgba": list(self.rgba) if self.rgba else None,
            "explanation": self.explanation,
            "sources_used": [dict(s) for s in self.sources_used],
            "error_reason": self.error_reason,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class Judgments:
    """Judgment outputs for standard or deep modes.

    Mutually exclusive: either standard or deep is populated, not both.
    """

    standard: StandardJudgment | None = None
    deep: tuple[DeepClaimResult, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        return {
            "standard": self.standard.to_payload() if self.standard else None,
            "deep": [result.to_payload() for result in self.deep],
        }


@dataclass(frozen=True, slots=True)
class CostSummary:
    """Cost breakdown from metering context."""

    credits: float
    llm_calls: int = 0
    search_calls: int = 0
    extract_calls: int = 0
    breakdown: dict[str, float] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "credits": self.credits,
            "llm_calls": self.llm_calls,
            "search_calls": self.search_calls,
            "extract_calls": self.extract_calls,
            "breakdown": dict(self.breakdown),
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class StandardFinalPayload:
    """Complete API response for standard mode.

    Frontend expects these exact fields for backward compatibility.
    """

    text: str
    rgba: tuple[float, float, float, float]
    sources: tuple[dict[str, Any], ...]
    rationale: str
    analysis: str
    details: tuple[str, ...] = field(default_factory=tuple)
    anchor_claim: dict[str, Any] | None = None
    cost_summary: CostSummary | None = None
    credits: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "rgba": list(self.rgba),
            "sources": [dict(s) for s in self.sources],
            "rationale": self.rationale,
            "analysis": self.analysis,
            "details": list(self.details),
            "anchor_claim": dict(self.anchor_claim) if self.anchor_claim else None,
            "cost_summary": self.cost_summary.to_payload() if self.cost_summary else None,
            "credits": self.credits,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class DeepAnalysis:
    """Container for deep mode per-claim results."""

    claim_results: tuple[DeepClaimResult, ...] = field(default_factory=tuple)
    judged_count: int = 0
    error_count: int = 0
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "claim_results": [r.to_payload() for r in self.claim_results],
            "judged_count": self.judged_count,
            "error_count": self.error_count,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True, slots=True)
class DeepFinalPayload:
    """Complete API response for deep mode.

    No global RGBA — per-claim results only.
    """

    prepared_text: str
    claims: tuple[dict[str, Any], ...]
    deep_analysis: DeepAnalysis
    cost_summary: CostSummary | None = None
    credits: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "prepared_text": self.prepared_text,
            "claims": [dict(c) for c in self.claims],
            "deep_analysis": self.deep_analysis.to_payload(),
            "cost_summary": self.cost_summary.to_payload() if self.cost_summary else None,
            "credits": self.credits,
            "trace": dict(self.trace),
        }
