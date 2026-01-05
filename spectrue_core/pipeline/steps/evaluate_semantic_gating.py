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

"""EVOI-based gating for expensive pipeline steps.

This module implements Expected Value of Information (EVOI) gating using
a calibratable logistic model for P(need) estimation.

The model is:
    p_need = sigmoid(w0 + sum(wi * xi))

Where features xi are derived from evidence signals and weights wi are
explicit priors that can be updated from historical data.

Cost estimation uses metering data when available, with conservative
fallbacks based on pipeline averages.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.pipeline.contracts import GATES_KEY, GateDecision, Gates
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


# =============================================================================
# LOGISTIC MODEL WEIGHTS (calibratable priors)
# =============================================================================
# These are initial priors. In production, calibrate from historical data:
#   w_new = w_old + learning_rate * (observed - predicted) * x

@dataclass(frozen=True)
class StanceWeights:
    """Logistic regression weights for P(stance needed)."""
    intercept: float = -1.5  # Prior: base probability ~18%
    log_sources: float = 0.4  # More sources → more value
    unlabeled_ratio: float = 0.8  # Unlabeled sources → need annotation
    low_tier_ratio: float = 0.5  # Low quality → need disambiguation
    log_claims: float = 0.3  # Multiple claims → more complexity
    quote_sparsity: float = 0.4  # Few quotes → stance helps


@dataclass(frozen=True)
class ClusterWeights:
    """Logistic regression weights for P(cluster needed)."""
    intercept: float = -2.0  # Prior: base probability ~12%
    log_sources: float = 0.5  # Many sources → need grouping
    log_claims: float = 0.6  # Multiple claims → per-claim grouping


# Default weight instances (can be overridden for A/B testing or calibration)
DEFAULT_STANCE_WEIGHTS = StanceWeights()
DEFAULT_CLUSTER_WEIGHTS = ClusterWeights()


# =============================================================================
# UTILITY ESTIMATION (calibratable)
# =============================================================================
# Delta utility: expected improvement if step changes the verdict
# These should be estimated from historical data: E[ΔU | Y=1]

STANCE_DELTA_UTILITY = 0.12  # Prior: 12% improvement in explainability
CLUSTER_DELTA_UTILITY = 0.08  # Prior: 8% improvement in source organization


# =============================================================================
# COST ESTIMATION
# =============================================================================

def _estimate_stance_cost(
    n_sources: int,
    n_claims: int,
    ledger: dict[str, Any] | None,
) -> float:
    """Estimate stance annotation cost from metering or model.
    
    If ledger has historical stance costs, use exponential moving average.
    Otherwise, use linear model: base + k1*sources + k2*claims.
    """
    # Try to get from metering history
    if ledger:
        stance_history = ledger.get("stance_costs", [])
        if stance_history:
            # Exponential moving average of last 5
            alpha = 0.3
            ema = stance_history[-1]
            for cost in reversed(stance_history[-5:-1]):
                ema = alpha * cost + (1 - alpha) * ema
            return ema
    
    # Linear cost model (calibrated from average pipeline runs)
    base_cost = 0.15
    per_source_cost = 0.03
    per_claim_cost = 0.05
    return base_cost + per_source_cost * n_sources + per_claim_cost * n_claims


def _estimate_cluster_cost(
    n_sources: int,
    n_claims: int,
    ledger: dict[str, Any] | None,
) -> float:
    """Estimate clustering cost from metering or model."""
    if ledger:
        cluster_history = ledger.get("cluster_costs", [])
        if cluster_history:
            alpha = 0.3
            ema = cluster_history[-1]
            for cost in reversed(cluster_history[-5:-1]):
                ema = alpha * cost + (1 - alpha) * ema
            return ema
    
    # Linear cost model
    base_cost = 0.10
    per_source_cost = 0.02
    per_claim_cost = 0.03
    return base_cost + per_source_cost * n_sources + per_claim_cost * n_claims


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@dataclass
class StanceFeatures:
    """Features for stance P(need) model."""
    log_sources: float
    unlabeled_ratio: float
    low_tier_ratio: float
    log_claims: float
    quote_sparsity: float
    
    @classmethod
    def from_evidence(
        cls,
        claims: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> "StanceFeatures":
        n_sources = len(sources)
        n_claims = len(claims)
        
        n_with_stance = sum(1 for s in sources if s.get("stance"))
        n_high_tier = sum(1 for s in sources if (s.get("tier") or "").upper() in {"A", "A'", "B"})
        n_with_quote = sum(1 for s in sources if s.get("quote"))
        
        return cls(
            log_sources=math.log1p(n_sources),
            unlabeled_ratio=1.0 - (n_with_stance / max(n_sources, 1)),
            low_tier_ratio=1.0 - (n_high_tier / max(n_sources, 1)),
            log_claims=math.log1p(n_claims),
            quote_sparsity=1.0 - (n_with_quote / max(n_sources, 1)),
        )


@dataclass
class ClusterFeatures:
    """Features for cluster P(need) model."""
    log_sources: float
    log_claims: float
    
    @classmethod
    def from_evidence(
        cls,
        claims: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> "ClusterFeatures":
        return cls(
            log_sources=math.log1p(len(sources)),
            log_claims=math.log1p(len(claims)),
        )


# =============================================================================
# LOGISTIC MODEL
# =============================================================================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _compute_stance_p_need(
    features: StanceFeatures,
    weights: StanceWeights = DEFAULT_STANCE_WEIGHTS,
) -> tuple[float, list[str]]:
    """Compute P(stance needed) using logistic model."""
    logit = (
        weights.intercept
        + weights.log_sources * features.log_sources
        + weights.unlabeled_ratio * features.unlabeled_ratio
        + weights.low_tier_ratio * features.low_tier_ratio
        + weights.log_claims * features.log_claims
        + weights.quote_sparsity * features.quote_sparsity
    )
    p_need = _sigmoid(logit)
    
    # Feature contributions for tracing
    reasons = [
        f"log_sources:{features.log_sources:.2f}",
        f"unlabeled:{features.unlabeled_ratio:.2f}",
        f"low_tier:{features.low_tier_ratio:.2f}",
        f"logit:{logit:.2f}",
    ]
    return p_need, reasons


def _compute_cluster_p_need(
    features: ClusterFeatures,
    weights: ClusterWeights = DEFAULT_CLUSTER_WEIGHTS,
) -> tuple[float, list[str]]:
    """Compute P(cluster needed) using logistic model."""
    logit = (
        weights.intercept
        + weights.log_sources * features.log_sources
        + weights.log_claims * features.log_claims
    )
    p_need = _sigmoid(logit)
    
    reasons = [
        f"log_sources:{features.log_sources:.2f}",
        f"log_claims:{features.log_claims:.2f}",
        f"logit:{logit:.2f}",
    ]
    return p_need, reasons


# =============================================================================
# GATE DECISIONS
# =============================================================================

def compute_stance_gate(
    claims: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    ledger: dict[str, Any] | None = None,
    cost_weight: float = 1.0,
) -> GateDecision:
    """Compute EVOI-based gate decision for stance annotation."""
    features = StanceFeatures.from_evidence(claims, sources)
    p_need, reasons = _compute_stance_p_need(features)
    
    expected_cost = _estimate_stance_cost(len(sources), len(claims), ledger)
    expected_gain = p_need * STANCE_DELTA_UTILITY
    threshold = expected_cost * cost_weight
    
    enabled = expected_gain > threshold
    
    return GateDecision(
        enabled=enabled,
        p_need=p_need,
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


def compute_cluster_gate(
    claims: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    stance_enabled: bool,
    ledger: dict[str, Any] | None = None,
    cost_weight: float = 1.0,
) -> GateDecision:
    """Compute EVOI-based gate decision for evidence clustering."""
    # Cluster only makes sense if stance was computed
    if not stance_enabled:
        return GateDecision(
            enabled=False,
            p_need=0.0,
            expected_gain=0.0,
            expected_cost=0.0,
            threshold=0.0,
            reasons=("stance_disabled",),
        )
    
    features = ClusterFeatures.from_evidence(claims, sources)
    p_need, reasons = _compute_cluster_p_need(features)
    
    expected_cost = _estimate_cluster_cost(len(sources), len(claims), ledger)
    expected_gain = p_need * CLUSTER_DELTA_UTILITY
    threshold = expected_cost * cost_weight
    
    enabled = expected_gain > threshold
    
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
class EvaluateSemanticGatingStep:
    """Evaluate semantic rejection policy and compute EVOI gates.

    This step performs two distinct functions:
    1. REJECTION: Block unverifiable content (legacy semantic gating)
    2. GATING: Compute EVOI-based gates for expensive steps

    Output:
        ctx.extras["gates"]: Gates contract with stance/cluster decisions
    """

    agent: Any
    name: str = "semantic_gating"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Evaluate semantic gating and compute gates."""
        try:
            claims = ctx.claims or []
            sources = ctx.sources or []
            
            # Get metering ledger if available
            ledger = ctx.get_extra("ledger")

            # ==================== PART 1: REJECTION POLICY ====================
            if hasattr(self.agent, "evaluate_semantic_gating"):
                candidates = ctx.get_extra("search_candidates", [])
                claims_to_check = candidates if candidates else ctx.get_extra("eligible_claims", claims)

                if claims_to_check:
                    is_allowed = await self.agent.evaluate_semantic_gating(claims_to_check)

                    if not is_allowed:
                        Trace.event("semantic_gating.rejected")
                        ctx = ctx.set_extra("search_candidates", [])
                        ctx = ctx.with_update(verdict={
                            "verified_score": 0.0,
                            "rationale": "Content validation rejected by semantic gating policy.",
                            "analysis": "Rejected by policy."
                        })
                        ctx = ctx.set_extra("gating_rejected", True)
                        # Rejected: all gates disabled
                        return ctx.set_extra(GATES_KEY, Gates())

            # ==================== PART 2: EVOI GATE COMPUTATION ====================
            stance_gate = compute_stance_gate(claims, sources, ledger)
            cluster_gate = compute_cluster_gate(claims, sources, stance_gate.enabled, ledger)

            gates = Gates(stance=stance_gate, cluster=cluster_gate)

            Trace.event(
                "semantic_gating.gates_computed",
                {
                    "stance_enabled": gates.is_stance_enabled(),
                    "stance_p_need": stance_gate.p_need,
                    "stance_gain": stance_gate.expected_gain,
                    "stance_cost": stance_gate.expected_cost,
                    "cluster_enabled": gates.is_cluster_enabled(),
                    "cluster_p_need": cluster_gate.p_need,
                },
            )

            return ctx.set_extra(GATES_KEY, gates)

        except Exception as e:
            logger.warning("[SemanticGatingStep] Failure: %s", e)
            # FALLBACK: DISABLE expensive steps (budget-safe)
            # Use prior p_need = 0.1 (conservative)
            fallback_gate = GateDecision(
                enabled=False,
                p_need=0.1,  # Conservative prior
                expected_gain=0.0,
                expected_cost=0.0,
                threshold=0.0,
                reasons=("fallback_on_error",),
            )
            return ctx.set_extra(GATES_KEY, Gates(stance=fallback_gate, cluster=fallback_gate))


