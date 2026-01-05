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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.contracts import GATES_KEY, GateDecision, Gates
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


# EVOI decision thresholds (expected_gain > expected_cost * cost_weight)
DEFAULT_COST_WEIGHT = 1.0
STANCE_COST_CREDITS = 0.5  # Approximate cost of stance annotation
CLUSTER_COST_CREDITS = 0.3  # Approximate cost of clustering


def _compute_stance_gate(
    claims: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> GateDecision:
    """Compute EVOI-based gate decision for stance annotation.

    Stance annotation is valuable when:
    - Multiple sources exist (otherwise nothing to annotate)
    - Sources lack stance labels already
    - High claim count (more to track)
    - Low source quality (need to disambiguate)
    """
    reasons: list[str] = []

    # Feature extraction
    n_claims = len(claims)
    n_sources = len(sources)
    n_with_stance = sum(1 for s in sources if s.get("stance"))
    n_high_tier = sum(1 for s in sources if (s.get("tier") or "").upper() in {"A", "A'", "B"})
    n_with_quote = sum(1 for s in sources if s.get("quote"))

    # P(need) based on signals
    p_need = 0.0

    # More sources = more value from stance disambiguation
    if n_sources > 5:
        p_need += 0.3
        reasons.append(f"many_sources:{n_sources}")
    elif n_sources > 2:
        p_need += 0.15
        reasons.append(f"some_sources:{n_sources}")

    # Sources without stance = annotation needed
    unlabeled_ratio = 1.0 - (n_with_stance / max(n_sources, 1))
    if unlabeled_ratio > 0.5:
        p_need += 0.2
        reasons.append(f"unlabeled_sources:{unlabeled_ratio:.2f}")

    # Low quality sources = need disambiguation
    low_tier_ratio = 1.0 - (n_high_tier / max(n_sources, 1))
    if low_tier_ratio > 0.6:
        p_need += 0.15
        reasons.append(f"low_tier_sources:{low_tier_ratio:.2f}")

    # Multiple claims = more complexity, stance helps
    if n_claims > 1:
        p_need += 0.1
        reasons.append(f"multiple_claims:{n_claims}")

    # Expected gain (utility improvement if stance helps)
    delta_utility = 0.1  # Baseline improvement in explainability
    if n_with_quote < n_sources * 0.3:
        delta_utility += 0.1  # More value when quotes are sparse
        reasons.append("few_quotes")

    expected_gain = p_need * delta_utility
    expected_cost = STANCE_COST_CREDITS
    threshold = expected_cost * DEFAULT_COST_WEIGHT

    enabled = expected_gain > threshold

    return GateDecision(
        enabled=enabled,
        p_need=min(p_need, 1.0),
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


def _compute_cluster_gate(
    claims: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    stance_enabled: bool,
) -> GateDecision:
    """Compute EVOI-based gate decision for evidence clustering.

    Clustering is valuable when:
    - Stance annotation is enabled (clustering uses stance)
    - High source count (need grouping)
    - Multiple claims (need per-claim assignment)
    """
    reasons: list[str] = []

    n_claims = len(claims)
    n_sources = len(sources)

    p_need = 0.0

    # Cluster only makes sense if stance was computed
    if not stance_enabled:
        return GateDecision(
            enabled=False,
            p_need=0.0,
            expected_gain=0.0,
            expected_cost=CLUSTER_COST_CREDITS,
            threshold=CLUSTER_COST_CREDITS * DEFAULT_COST_WEIGHT,
            reasons=("stance_disabled",),
        )

    # Many sources = clustering helps organize
    if n_sources > 8:
        p_need += 0.4
        reasons.append(f"many_sources:{n_sources}")
    elif n_sources > 4:
        p_need += 0.2
        reasons.append(f"some_sources:{n_sources}")

    # Multiple claims = per-claim grouping valuable
    if n_claims > 2:
        p_need += 0.3
        reasons.append(f"multiple_claims:{n_claims}")
    elif n_claims > 1:
        p_need += 0.15
        reasons.append(f"few_claims:{n_claims}")

    delta_utility = 0.08  # Baseline clustering benefit
    expected_gain = p_need * delta_utility
    expected_cost = CLUSTER_COST_CREDITS
    threshold = expected_cost * DEFAULT_COST_WEIGHT

    enabled = expected_gain > threshold

    return GateDecision(
        enabled=enabled,
        p_need=min(p_need, 1.0),
        expected_gain=expected_gain,
        expected_cost=expected_cost,
        threshold=threshold,
        reasons=tuple(reasons),
    )


@dataclass
class EvaluateSemanticGatingStep:
    """Evaluate semantic gating and compute EVOI gates for expensive steps.

    This step performs two functions:
    1. Reject unverifiable content (legacy semantic gating)
    2. Compute EVOI-based gates for stance/cluster steps

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

            # ==================== PART 1: LEGACY SEMANTIC GATING ====================
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
                        # Return early with default gates (all disabled)
                        return ctx.set_extra(GATES_KEY, Gates())

            # ==================== PART 2: EVOI GATE COMPUTATION ====================
            stance_gate = _compute_stance_gate(claims, sources)
            cluster_gate = _compute_cluster_gate(claims, sources, stance_gate.enabled)

            gates = Gates(stance=stance_gate, cluster=cluster_gate)

            Trace.event(
                "semantic_gating.gates_computed",
                {
                    "stance_enabled": gates.is_stance_enabled(),
                    "cluster_enabled": gates.is_cluster_enabled(),
                    "stance_reasons": list(stance_gate.reasons),
                    "cluster_reasons": list(cluster_gate.reasons),
                },
            )

            return ctx.set_extra(GATES_KEY, gates)

        except Exception as e:
            logger.warning("[SemanticGatingStep] Failure: %s", e)
            # On failure, default to gates enabled (safe fallback)
            default_gate = GateDecision(
                enabled=True,
                p_need=1.0,
                expected_gain=1.0,
                expected_cost=0.0,
                threshold=0.0,
                reasons=("fallback_on_error",),
            )
            return ctx.set_extra(GATES_KEY, Gates(stance=default_gate, cluster=default_gate))

