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

"""EVOI gating step that runs AFTER evidence collection.

This step computes gates for expensive analysis steps (stance/cluster)
using proper signals from EvidenceIndex, not raw ctx.sources.

The gating decision uses:
1. Evidence quality signals (tier distribution, quote presence)
2. Uncertainty estimation from evidence (support/refute ratio entropy)
3. Metering history for cost estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from spectrue_core.pipeline.contracts import (
    EVIDENCE_INDEX_KEY,
    GATES_KEY,
    EvidenceIndex,
    GateDecision,
    Gates,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace
from spectrue_core.domain.evidence.gating import ClusterWeights, StanceWeights
from spectrue_core.use_cases.evidence.gating import compute_gates

logger = logging.getLogger(__name__)


DEFAULT_STANCE_WEIGHTS = StanceWeights()
DEFAULT_CLUSTER_WEIGHTS = ClusterWeights()

@dataclass
class EvidenceGatingStep:
    """Compute EVOI gates for expensive analysis steps.
    
    This step runs AFTER evidence collection and reads from EvidenceIndex,
    not raw ctx.sources. This ensures gating decisions use proper retrieval signals.
    
    Input:
        ctx.extras[EVIDENCE_INDEX_KEY]: Evidence collected per claim
        
    Output:
        ctx.extras[GATES_KEY]: Gates with stance/cluster decisions
    """
    
    name: str = "evidence_gating"
    weight: float = 1.0
    
    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            # Skip if already rejected
            if ctx.get_extra("gating_rejected"):
                return ctx.set_extra(GATES_KEY, Gates())
            
            # Get evidence index (proper signal source)
            evidence_index: EvidenceIndex | None = ctx.get_extra(EVIDENCE_INDEX_KEY)
            if evidence_index is None:
                # No evidence collected, disable expensive steps
                Trace.event("evidence_gating.no_evidence")
                fallback = GateDecision(
                    enabled=False, p_need=0.0, expected_gain=0.0,
                    expected_cost=0.0, threshold=0.0, reasons=("no_evidence",)
                )
                return ctx.set_extra(GATES_KEY, Gates(stance=fallback, cluster=fallback))
            
            claims = ctx.claims or []
            
            ledger = ctx.get_extra("ledger")
            features, uncertainty, stance_gate_payload, cluster_gate_payload = compute_gates(
                evidence_index=evidence_index,
                claims=claims,
                ledger=ledger,
                stance_weights=DEFAULT_STANCE_WEIGHTS,
                cluster_weights=DEFAULT_CLUSTER_WEIGHTS,
            )

            stance_gate = GateDecision(
                enabled=stance_gate_payload.enabled,
                p_need=stance_gate_payload.p_need,
                expected_gain=stance_gate_payload.expected_gain,
                expected_cost=stance_gate_payload.expected_cost,
                threshold=stance_gate_payload.threshold,
                reasons=stance_gate_payload.reasons,
            )
            cluster_gate = GateDecision(
                enabled=cluster_gate_payload.enabled,
                p_need=cluster_gate_payload.p_need,
                expected_gain=cluster_gate_payload.expected_gain,
                expected_cost=cluster_gate_payload.expected_cost,
                threshold=cluster_gate_payload.threshold,
                reasons=cluster_gate_payload.reasons,
            )
            
            gates = Gates(stance=stance_gate, cluster=cluster_gate)
            
            Trace.event(
                "evidence_gating.computed",
                {
                    "stance_enabled": gates.is_stance_enabled(),
                    "stance_p_need": stance_gate.p_need,
                    "stance_gain": stance_gate.expected_gain,
                    "stance_cost": stance_gate.expected_cost,
                    "cluster_enabled": gates.is_cluster_enabled(),
                    "uncertainty": uncertainty,
                    "n_evidence": features["n_evidence"],
                },
            )
            
            return ctx.set_extra(GATES_KEY, gates)
            
        except Exception as e:
            logger.warning("[EvidenceGatingStep] Failure: %s", e)
            # FALLBACK: DISABLE (budget-safe)
            fallback = GateDecision(
                enabled=False,
                p_need=0.1,
                expected_gain=0.0,
                expected_cost=0.0,
                threshold=0.0,
                reasons=("fallback_on_error",),
            )
            return ctx.set_extra(GATES_KEY, Gates(stance=fallback, cluster=fallback))
