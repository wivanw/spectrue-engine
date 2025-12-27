# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Reason code taxonomy for traceable routing decisions.

Codes are versioned to keep run-to-run comparisons stable as routing evolves.
"""

from __future__ import annotations

from dataclasses import dataclass

REASON_CODE_VERSION = "v1"


@dataclass(frozen=True)
class ReasonCodeSpec:
    code: str
    label: str
    phase: str
    action: str

    def qualified(self) -> str:
        return f"{REASON_CODE_VERSION}.{self.code}"


class ReasonCodes:
    """Canonical reason code definitions used in ledgers and traces."""

    POLICY_SKIP = ReasonCodeSpec(
        code="policy.skip",
        label="Policy decided to skip search",
        phase="query_build",
        action="skip",
    )
    POLICY_CHEAP = ReasonCodeSpec(
        code="policy.cheap",
        label="Policy decided to run cheap mode",
        phase="query_build",
        action="route",
    )
    POLICY_FULL = ReasonCodeSpec(
        code="policy.full",
        label="Policy decided to run full mode",
        phase="query_build",
        action="route",
    )
    BUDGET_LOW = ReasonCodeSpec(
        code="budget.low",
        label="Budget capped by low worthiness",
        phase="query_build",
        action="budget",
    )
    BUDGET_HIGH = ReasonCodeSpec(
        code="budget.high",
        label="Budget expanded by high worthiness",
        phase="query_build",
        action="budget",
    )
    BUDGET_MEDIUM = ReasonCodeSpec(
        code="budget.medium",
        label="Budget set for medium worthiness",
        phase="query_build",
        action="budget",
    )
    RETRIEVAL_CONF_HIGH = ReasonCodeSpec(
        code="retrieval.confidence.high",
        label="Retrieval confidence high",
        phase="retrieval",
        action="stop_early",
    )
    RETRIEVAL_CONF_LOW = ReasonCodeSpec(
        code="retrieval.confidence.low",
        label="Retrieval confidence low",
        phase="retrieval",
        action="correct",
    )
    RETRIEVAL_CONF_MED = ReasonCodeSpec(
        code="retrieval.confidence.medium",
        label="Retrieval confidence medium",
        phase="retrieval",
        action="continue",
    )
    RETRIEVAL_CORRECTION = ReasonCodeSpec(
        code="retrieval.correction",
        label="Corrective retrieval action applied",
        phase="retrieval",
        action="correct",
    )
    RETRIEVAL_STOP_EARLY = ReasonCodeSpec(
        code="retrieval.stop_early",
        label="Stop early after sufficient evidence",
        phase="retrieval",
        action="stop_early",
    )
    RETRIEVAL_STOP_MAX_HOPS = ReasonCodeSpec(
        code="retrieval.stop_max_hops",
        label="Stop retrieval after reaching max hops",
        phase="retrieval",
        action="stop",
    )
    RETRIEVAL_STOP_FOLLOWUP_FAILED = ReasonCodeSpec(
        code="retrieval.stop_followup_failed",
        label="Stop retrieval due to missing follow-up query",
        phase="retrieval",
        action="stop",
    )
    EVIDENCE_SUFFICIENT = ReasonCodeSpec(
        code="evidence.sufficient",
        label="Evidence sufficiency met",
        phase="evidence_eval",
        action="stop_early",
    )
    GRAPH_PROPAGATION_DOC = ReasonCodeSpec(
        code="graph.doc_alignment",
        label="Graph documentation aligned",
        phase="graph",
        action="document",
    )
    CLAIM_DEFERRED = ReasonCodeSpec(
        code="target.deferred",
        label="Claim deferred from search (not in top-K targets)",
        phase="query_build",
        action="defer",
    )
