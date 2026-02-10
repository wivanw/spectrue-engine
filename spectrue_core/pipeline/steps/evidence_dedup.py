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

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace
from spectrue_core.use_cases.evidence.dedup import apply_dedup


@dataclass
class EvidenceDedupStep(Step):
    """
    Compute exact-dup and near-dup fingerprints for EvidenceItems.
    - publisher_id: normalized domain
    - content_hash: sha256(normalized payload)
    - similar_cluster_id: simhash bucket id
    """
    weight: float = 1.0

    name: str = "evidence_dedup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        if not sources:
            return ctx

        stats = apply_dedup(sources=sources)

        Trace.event(
            "evidence_dedup.completed",
            stats,
        )
        return ctx
