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

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class TargetSelectionStep:
    """
    Select which claims get actual searches.

    Implements the critical gate that prevents per-claim search explosion.
    Only top-K key claims trigger Tavily searches.

    Context Input:
        - extras: eligible_claims, graph_result, anchor_claim_id
        - mode (for budget_class derivation)

    Context Output:
        - extras: target_claims, deferred_claims, evidence_sharing
    """

    name: str = "target_selection"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Select verification targets."""
        from spectrue_core.verification.target_selection import select_verification_targets

        try:
            eligible_claims = ctx.get_extra("eligible_claims", ctx.claims)
            graph_result = ctx.get_extra("graph_result")
            anchor_claim_id = ctx.get_extra("anchor_claim_id")
            
            # M118/T001: Skip if valid verdict already exists (e.g. Oracle Jackpot)
            if ctx.get_extra("oracle_hit") or ctx.get_extra("final_result"):
                return ctx.set_extra("target_claims", []).set_extra("search_candidates", [])

            # Derive budget_class from mode
            budget_class = {
                "basic": "minimal",
                "advanced": "deep",
            }.get(ctx.mode.search_depth, "standard")

            # For normal mode, anchor must be in targets
            anchor_for_selection = anchor_claim_id if ctx.mode.name == "normal" else None

            # DEEP MODE: Verify ALL claims (no target selection limits)
            if ctx.mode.name == "deep":
                # All claims are targets in deep mode
                target_claims = list(eligible_claims)
                deferred_claims = []
                evidence_sharing = {}
                target_reasons = {c.get("id"): "deep_mode_all_verified" for c in target_claims}

                Trace.event(
                    "target_selection.deep_mode_all_claims",
                    {
                        "mode": ctx.mode.name,
                        "claims_count": len(target_claims),
                        "claim_ids": [c.get("id") for c in target_claims],
                    },
                )
            else:
                result = select_verification_targets(
                    claims=eligible_claims,
                    # max_targets removed: let Bayesian EVOI model compute
                    graph_result=graph_result,
                    budget_class=budget_class,
                    anchor_claim_id=anchor_for_selection,
                )
                target_claims = result.targets
                deferred_claims = result.deferred
                evidence_sharing = result.evidence_sharing
                target_reasons = result.reasons

            Trace.event(
                    "target_selection.step_completed",
                    {
                        "targets_count": len(target_claims),
                        "deferred_count": len(deferred_claims),
                    },
                )

            return (
                ctx.set_extra("target_claims", target_claims)
                .set_extra("search_candidates", target_claims) # Alias for SearchFlow
                .set_extra("deferred_claims", deferred_claims)
                .set_extra("evidence_sharing", evidence_sharing)
                .set_extra("target_reasons", target_reasons)
            )

        except Exception as e:
            logger.exception("[TargetSelectionStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
