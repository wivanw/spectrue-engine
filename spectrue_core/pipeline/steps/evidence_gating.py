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
import math
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.contracts import (
    EVIDENCE_INDEX_KEY,
    GATES_KEY,
    EvidenceIndex,
    GateDecision,
    Gates,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


# =============================================================================
# LOGISTIC MODEL WEIGHTS (calibratable priors)
# =============================================================================

@dataclass(frozen=True)
class StanceWeights:
    """Logistic regression weights for P(stance needed)."""
    intercept: float = -1.5
    log_evidence: float = 0.4
    unlabeled_ratio: float = 0.8
    low_tier_ratio: float = 0.5
    log_claims: float = 0.3
    uncertainty: float = 1.2  # Weight for evidence uncertainty


@dataclass(frozen=True)
class ClusterWeights:
    """Logistic regression weights for P(cluster needed)."""
    intercept: float = -2.0
    log_evidence: float = 0.5
    log_claims: float = 0.6


DEFAULT_STANCE_WEIGHTS = StanceWeights()
DEFAULT_CLUSTER_WEIGHTS = ClusterWeights()


# =============================================================================
# UNCERTAINTY ESTIMATION
# =============================================================================

def _compute_evidence_uncertainty(evidence_index: EvidenceIndex) -> float:
    """Compute uncertainty from evidence stance distribution.
    
    Returns entropy of support/refute/neutral distribution.
    High entropy = high uncertainty = stance annotation more valuable.
    """
    n_support = 0
    n_refute = 0
    n_neutral = 0
    
    for pack in evidence_index.by_claim_id.values():
        for item in pack.items:
            stance = (item.stance or "").upper()
            if stance == "SUPPORT":
                n_support += 1
            elif stance == "REFUTE":
                n_refute += 1
            else:
                n_neutral += 1
    
    total = n_support + n_refute + n_neutral
    if total == 0:
        return 1.0  # Maximum uncertainty when no evidence
    
    # Compute entropy (normalized to 0-1)
    probs = [n_support / total, n_refute / total, n_neutral / total]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p + 1e-10)
    
    # Normalize by max entropy (log 3)
    max_entropy = math.log(3)
    return entropy / max_entropy


def _compute_delta_utility(uncertainty: float, base_utility: float = 0.05) -> float:
    """Compute expected utility improvement from running a step.
    
    Higher uncertainty = higher potential improvement.
    """
    # k * uncertainty where k is calibratable
    k = 0.15
    return base_utility + k * uncertainty


# =============================================================================
# FEATURE EXTRACTION FROM EVIDENCE INDEX
# =============================================================================

def _extract_evidence_features(
    evidence_index: EvidenceIndex,
    claims: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract features from EvidenceIndex (not raw sources)."""
    total_items = 0
    n_with_stance = 0
    n_high_tier = 0
    n_with_quote = 0
    
    for pack in evidence_index.by_claim_id.values():
        for item in pack.items:
            total_items += 1
            if item.stance:
                n_with_stance += 1
            if (item.tier or "").upper() in {"A", "A'", "B"}:
                n_high_tier += 1
            if item.quote:
                n_with_quote += 1
    
    n_claims = len(claims)
    
    return {
        "log_evidence": math.log1p(total_items),
        "unlabeled_ratio": 1.0 - (n_with_stance / max(total_items, 1)),
        "low_tier_ratio": 1.0 - (n_high_tier / max(total_items, 1)),
        "log_claims": math.log1p(n_claims),
        "quote_sparsity": 1.0 - (n_with_quote / max(total_items, 1)),
        "n_evidence": total_items,
        "n_claims": n_claims,
    }


# =============================================================================
# COST ESTIMATION
# =============================================================================

def _get_ledger_history(ctx: PipelineContext, key: str) -> list[float]:
    """Safely extract cost history from ledger."""
    ledger = ctx.get_extra("ledger")
    if ledger is None:
        return []
    
    # Handle Ledger object or dict
    if hasattr(ledger, "get_history"):
        return ledger.get_history(key) or []
    if isinstance(ledger, dict):
        return ledger.get(key, [])
    
    return []


def _estimate_cost_ema(history: list[float], fallback: float) -> float:
    """Compute EMA of cost history, or fallback."""
    if not history:
        return fallback
    
    alpha = 0.3
    ema = history[-1]
    for cost in reversed(history[-5:-1]):
        ema = alpha * cost + (1 - alpha) * ema
    return ema


def _estimate_stance_cost(features: dict[str, float], ctx: PipelineContext) -> float:
    """Estimate stance cost from history or linear model."""
    history = _get_ledger_history(ctx, "stance_costs")
    if history:
        return _estimate_cost_ema(history, 0.3)
    
    # Linear model fallback
    n_evidence = features["n_evidence"]
    n_claims = features["n_claims"]
    return 0.15 + 0.03 * n_evidence + 0.05 * n_claims


def _estimate_cluster_cost(features: dict[str, float], ctx: PipelineContext) -> float:
    """Estimate cluster cost from history or linear model."""
    history = _get_ledger_history(ctx, "cluster_costs")
    if history:
        return _estimate_cost_ema(history, 0.2)
    
    n_evidence = features["n_evidence"]
    n_claims = features["n_claims"]
    return 0.10 + 0.02 * n_evidence + 0.03 * n_claims


# =============================================================================
# SIGMOID MODEL
# =============================================================================

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _compute_stance_gate(
    features: dict[str, float],
    uncertainty: float,
    ctx: PipelineContext,
    weights: StanceWeights = DEFAULT_STANCE_WEIGHTS,
    cost_weight: float = 1.0,
) -> GateDecision:
    """Compute stance gate using logistic model."""
    # Compute logit with feature contributions
    contributions = {
        "intercept": weights.intercept,
        "log_evidence": weights.log_evidence * features["log_evidence"],
        "unlabeled": weights.unlabeled_ratio * features["unlabeled_ratio"],
        "low_tier": weights.low_tier_ratio * features["low_tier_ratio"],
        "log_claims": weights.log_claims * features["log_claims"],
        "uncertainty": weights.uncertainty * uncertainty,
    }
    
    logit = sum(contributions.values())
    p_need = _sigmoid(logit)
    
    expected_cost = _estimate_stance_cost(features, ctx)
    delta_utility = _compute_delta_utility(uncertainty)
    expected_gain = p_need * delta_utility
    threshold = expected_cost * cost_weight
    
    enabled = expected_gain > threshold
    
    # Reasons with contributions (for explainability)
    reasons = [f"{k}:{v:.3f}" for k, v in contributions.items()]
    reasons.append(f"logit:{logit:.2f}")
    reasons.append(f"p_need:{p_need:.3f}")
    
    return GateDecision(
        enabled=enabled,
        p_need=p_need,
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


def _compute_cluster_gate(
    features: dict[str, float],
    stance_enabled: bool,
    ctx: PipelineContext,
    weights: ClusterWeights = DEFAULT_CLUSTER_WEIGHTS,
    cost_weight: float = 1.0,
) -> GateDecision:
    """Compute cluster gate using logistic model."""
    # Cluster requires stance
    if not stance_enabled:
        expected_cost = _estimate_cluster_cost(features, ctx)
        return GateDecision(
            enabled=False,
            p_need=0.0,
            expected_gain=0.0,
            expected_cost=expected_cost,  # Real cost for accounting
            threshold=expected_cost * cost_weight,
            reasons=("stance_disabled",),
        )
    
    contributions = {
        "intercept": weights.intercept,
        "log_evidence": weights.log_evidence * features["log_evidence"],
        "log_claims": weights.log_claims * features["log_claims"],
    }
    
    logit = sum(contributions.values())
    p_need = _sigmoid(logit)
    
    expected_cost = _estimate_cluster_cost(features, ctx)
    delta_utility = 0.08  # Base utility for clustering
    expected_gain = p_need * delta_utility
    threshold = expected_cost * cost_weight
    
    enabled = expected_gain > threshold
    
    reasons = [f"{k}:{v:.3f}" for k, v in contributions.items()]
    reasons.append(f"logit:{logit:.2f}")
    
    return GateDecision(
        enabled=enabled,
        p_need=p_need,
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


# =============================================================================
# STEP IMPLEMENTATION
# =============================================================================

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
            
            # Extract features from EvidenceIndex
            features = _extract_evidence_features(evidence_index, claims)
            
            # Compute uncertainty from evidence
            uncertainty = _compute_evidence_uncertainty(evidence_index)
            
            # Compute gates
            stance_gate = _compute_stance_gate(features, uncertainty, ctx)
            cluster_gate = _compute_cluster_gate(features, stance_gate.enabled, ctx)
            
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
