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
"""
Run-level usage ledger data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ReasonCode:
    code: str
    label: str | None = None
    phase: str | None = None
    action: str | None = None
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "phase": self.phase,
            "action": self.action,
            "count": int(self.count),
        }


@dataclass
class ReasonSummary:
    code: str
    count: int
    sc_cost: float = 0.0
    tc_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "count": int(self.count),
            "sc_cost": float(self.sc_cost),
            "tc_cost": float(self.tc_cost),
        }


@dataclass
class PhaseUsage:
    phase: str
    sc_cost: float = 0.0
    tc_cost: float = 0.0
    duration_ms: int = 0
    reason_codes: list[ReasonCode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "sc_cost": float(self.sc_cost),
            "tc_cost": float(self.tc_cost),
            "duration_ms": int(self.duration_ms),
            "reason_codes": [rc.to_dict() for rc in self.reason_codes],
        }


@dataclass
class PipelineCounts:
    claims_total: int = 0
    claims_eligible: int = 0
    queries_total: int = 0
    docs_total: int = 0
    evidence_units_total: int = 0
    llm_calls_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims_total": int(self.claims_total),
            "claims_eligible": int(self.claims_eligible),
            "queries_total": int(self.queries_total),
            "docs_total": int(self.docs_total),
            "evidence_units_total": int(self.evidence_units_total),
            "llm_calls_total": int(self.llm_calls_total),
        }


@dataclass
class BudgetAllocation:
    worthiness_tier: str
    max_queries: int
    max_docs: int
    max_escalations: int
    defer_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "worthiness_tier": self.worthiness_tier,
            "max_queries": int(self.max_queries),
            "max_docs": int(self.max_docs),
            "max_escalations": int(self.max_escalations),
            "defer_allowed": bool(self.defer_allowed),
        }


@dataclass
class RetrievalEvaluation:
    cycle_index: int
    relevance_score: float
    evidence_likeness_score: float
    source_quality_score: float
    retrieval_confidence: float
    action: str
    reason_code: ReasonCode | None = None
    expected_gain: float = 0.0
    expected_cost: float = 0.0
    value_per_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "relevance_score": float(self.relevance_score),
            "evidence_likeness_score": float(self.evidence_likeness_score),
            "source_quality_score": float(self.source_quality_score),
            "retrieval_confidence": float(self.retrieval_confidence),
            "action": self.action,
            "reason_code": self.reason_code.to_dict() if self.reason_code else None,
            "expected_gain": float(self.expected_gain),
            "expected_cost": float(self.expected_cost),
            "value_per_cost": float(self.value_per_cost),
        }


@dataclass
class ClaimLedgerEntry:
    claim_id: str
    policy_mode: str
    policy_reasons: list[ReasonCode] = field(default_factory=list)
    budget_allocation: BudgetAllocation | None = None
    retrieval_evaluations: list[RetrievalEvaluation] = field(default_factory=list)
    stop_reason: ReasonCode | None = None
    queries_used: int = 0
    docs_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "policy_mode": self.policy_mode,
            "policy_reasons": [rc.to_dict() for rc in self.policy_reasons],
            "budget_allocation": self.budget_allocation.to_dict() if self.budget_allocation else None,
            "retrieval_evaluations": [ev.to_dict() for ev in self.retrieval_evaluations],
            "stop_reason": self.stop_reason.to_dict() if self.stop_reason else None,
            "queries_used": int(self.queries_used),
            "docs_used": int(self.docs_used),
        }


@dataclass
class ClusterLedgerEntry:
    cluster_id: str
    claim_ids: list[str] = field(default_factory=list)
    shared_query_ids: list[str] = field(default_factory=list)
    budget_allocation: BudgetAllocation | None = None
    duplicate_query_savings: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "claim_ids": list(self.claim_ids),
            "shared_query_ids": list(self.shared_query_ids),
            "budget_allocation": self.budget_allocation.to_dict() if self.budget_allocation else None,
            "duplicate_query_savings": int(self.duplicate_query_savings),
        }


@dataclass
class RunLedger:
    run_id: str | None
    engine_version: str | None = None
    prompt_version: str | None = None
    search_strategy: str | None = None
    phase_usage: list[PhaseUsage] = field(default_factory=list)
    counts: PipelineCounts = field(default_factory=PipelineCounts)
    top_reason_codes: list[ReasonSummary] = field(default_factory=list)
    claim_entries: list[ClaimLedgerEntry] = field(default_factory=list)
    cluster_entries: list[ClusterLedgerEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "prompt_version": self.prompt_version,
            "search_strategy": self.search_strategy,
            "phase_usage": [pu.to_dict() for pu in self.phase_usage],
            "counts": self.counts.to_dict(),
            "top_reason_codes": [rc.to_dict() for rc in self.top_reason_codes],
            "claim_entries": [ce.to_dict() for ce in self.claim_entries],
            "cluster_entries": [ce.to_dict() for ce in self.cluster_entries],
            "created_at": self.created_at.isoformat(),
        }
