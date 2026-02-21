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

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.use_cases.claims.graph import build_claim_graph
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ClaimGraphStep:
    """
    Build claim graph for relationship analysis.

    Uses NetworkX to build graph of claim relationships,
    identify key claims, and compute centrality metrics.

    Context Input:
        - claims
        - extras: eligible_claims

    Context Output:
        - extras: graph_result, key_claims
    """

    claim_graph: Any  # ClaimGraphBuilder instance
    runtime_config: Any  # RuntimeConfig
    name: str = "claim_graph"
    weight: float = 12.0  # ~12.3s actual (graph build + LLM edge typing)

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Build claim graph."""
        try:
            eligible_claims = ctx.get_extra("eligible_claims", ctx.claims)

            # Skip if only one claim
            if len(eligible_claims) <= 1:
                Trace.event("claim_graph.skipped", {"reason": "single_claim"})
                key_ids = [c.get("id") for c in eligible_claims] if eligible_claims else []
                return ctx.set_extra("graph_result", None).set_extra("key_claim_ids", key_ids)
                
            # Adaptive execution gate (T031)
            # If the budget allows all claims to be selected, graph ranking adds zero value
            # and only incurs latency and LLM costs.
            cfg = getattr(self.runtime_config, "claim_graph", None)
            if cfg:
                budget = getattr(cfg, "selection_budget", None)
                default_cost = getattr(cfg, "default_claim_cost", 1.0)
                if budget is not None and budget > 0:
                    worst_case_cost = len(eligible_claims) * max(default_cost, 1.0)
                    if worst_case_cost <= budget:
                        Trace.event("claim_graph.skipped", {"reason": "budget_covers_all_claims"})
                        key_ids = [c.get("id") or f"c{i}" for i, c in enumerate(eligible_claims)]
                        return ctx.set_extra("graph_result", None).set_extra("key_claim_ids", key_ids)

            progress_callback = ctx.get_extra("progress_callback")
            
            result = await build_claim_graph(
                self.claim_graph,
                claims=eligible_claims,
                runtime_config=self.runtime_config,
                progress_callback=progress_callback,
            )

            Trace.event(
                "claim_graph.completed",
                {
                    "claims_in_graph": len(eligible_claims),
                    "key_claims_count": len(result.key_claim_ids),
                },
            )

            return ctx.set_extra("graph_result", result.graph_result).set_extra(
                "key_claim_ids", result.key_claim_ids
            )

        except Exception as e:
            logger.warning("[ClaimGraphStep] Non-fatal failure: %s", e)
            Trace.event("claim_graph.error", {"error": str(e)})
            return ctx.set_extra("graph_result", None)
