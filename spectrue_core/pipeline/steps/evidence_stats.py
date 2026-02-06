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

from spectrue_core.use_cases.evidence.stats import compute_stats
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url


@dataclass
class EvidenceStatsStep(Step):
    """
    Build per-claim EvidenceStats from already selected sources.
    This decouples explainability (A) from BudgetState and fetch/extract counts.
    """
    weight: float = 1.0
    name: str = "evidence_stats"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        # Prefer evidence_by_claim if present (built by spillover steps)
        by_claim = ctx.get_extra("evidence_by_claim")
        if not isinstance(by_claim, dict):
            by_claim = None

        cluster_map = ctx.get_extra("cluster_map") or {}
        cluster_sufficiency = ctx.get_extra("cluster_sufficiency") or {}
        result = compute_stats(
            sources=sources,
            claims=claims,
            cluster_map=cluster_map,
            cluster_sufficiency=cluster_sufficiency,
            normalize_url=normalize_url,
            evidence_by_claim=by_claim,
        )

        Trace.event(
            "evidence_stats.completed",
            {
                "claims": len(result.by_claim),
                "avg_sources": result.avg_sources,
                "avg_A_det": result.avg_explainability,
            },
        )

        return ctx.set_extra("evidence_stats_by_claim", result.by_claim)
