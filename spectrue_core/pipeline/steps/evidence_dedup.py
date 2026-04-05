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
    Compute exact-dup and near-dup fingerprints for EvidenceItems,
    then remove exact duplicates (keep best source per content_hash group).

    Runs in ALL analysis modes.
    """
    weight: float = 1.0

    name: str = "evidence_dedup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        sources = ctx.sources or []
        if not sources:
            return ctx

        filtered, stats = apply_dedup(sources=sources, filter_duplicates=True)

        Trace.event("evidence_dedup.completed", stats)

        if stats.get("removed", 0) > 0:
            ctx = ctx.with_update(sources=filtered)

            # Update evidence_by_claim if present
            by_claim = ctx.get_extra("evidence_by_claim")
            if isinstance(by_claim, dict):
                kept_urls = {s.get("url") for s in filtered if s.get("url")}
                updated_by_claim = {}
                for cid, items in by_claim.items():
                    if isinstance(items, list):
                        updated_by_claim[cid] = [
                            i for i in items
                            if not i.get("url") or i.get("url") in kept_urls
                        ]
                    else:
                        updated_by_claim[cid] = items
                ctx = ctx.set_extra("evidence_by_claim", updated_by_claim)

        return ctx
