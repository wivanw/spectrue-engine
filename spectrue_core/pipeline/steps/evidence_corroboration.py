# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass

from spectrue_core.use_cases.evidence.corroboration import compute_corroboration
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace


@dataclass
class EvidenceCorroborationStep(Step):
    """
    Compute per-claim corroboration counters using:
    - precise confirmations: unique publishers with SUPPORT/REFUTE and direct anchors
    - corroboration confirmations: unique similar clusters with SUPPORT/REFUTE (any anchor)
    - exact duplicate count (informational)
    """
    weight: float = 1.0

    name: str = "evidence_corroboration"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        # Prefer evidence_by_claim
        by_claim = ctx.get_extra("evidence_by_claim")
        if not isinstance(by_claim, dict):
            by_claim = None

        result = compute_corroboration(
            sources=sources,
            claims=claims,
            evidence_by_claim=by_claim,
        )

        Trace.event(
            "evidence_corroboration.completed",
            {
                "claims": len(result.by_claim),
                "avg_precise_support": result.avg_precise_support,
                "avg_corr_support": result.avg_corr_support,
            },
        )

        return ctx.set_extra("corroboration_by_claim", result.by_claim)
