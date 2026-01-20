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

"""Semantic rejection step for unverifiable content.

This step runs EARLY in the pipeline (before retrieval) and performs:
1. Semantic gating rejection policy (block unverifiable content)

Does NOT compute EVOI gates — that's done by EvidenceGatingStep after evidence collection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class EvaluateSemanticGatingStep:
    """Evaluate semantic rejection policy for unverifiable content.

    This is a rejection policy step, NOT an EVOI gating step.
    If content is rejected, sets gating_rejected flag for downstream steps.

    EVOI gates for stance/cluster are computed later by EvidenceGatingStep.
    """

    agent: Any
    name: str = "semantic_gating"
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Evaluate semantic rejection policy."""
        try:
            if not hasattr(self.agent, "evaluate_semantic_gating"):
                return ctx

            candidates = ctx.get_extra("search_candidates", [])
            claims_to_check = candidates if candidates else ctx.get_extra(
                "eligible_claims", ctx.claims or []
            )

            if not claims_to_check:
                return ctx

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

            return ctx

        except Exception as e:
            logger.warning("[SemanticGatingStep] Failure: %s", e)
            return ctx
