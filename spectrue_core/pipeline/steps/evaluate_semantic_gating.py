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
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class EvaluateSemanticGatingStep:
    """
    Evaluate semantic gating policy (e.g. reject unverifiable content).
    
    If rejected, clears search candidates and sets rejection verdict.
    """
    
    agent: Any
    name: str = "semantic_gating"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Evaluate semantic gating."""
        try:
            # Check if enabled in runtime (optional)
            # Legacy logic checks if agent has the method
            if not hasattr(self.agent, "evaluate_semantic_gating"):
                 return ctx

            # Use search candidates if available (post-target-selection)
            # Or eligible claims?
            # Legacy used "claim_units" which are derived from claims.
            # Passing list of claim dicts is safer for broad compatibility.
            candidates = ctx.get_extra("search_candidates", [])
            claims_to_check = candidates if candidates else ctx.get_extra("eligible_claims", ctx.claims)

            if not claims_to_check:
                return ctx

            is_allowed = await self.agent.evaluate_semantic_gating(claims_to_check)
            
            if not is_allowed:
                 Trace.event("semantic_gating.rejected")
                 # Reject: clear candidates to stop search
                 ctx = ctx.set_extra("search_candidates", [])
                 # Set rejected verdict
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
