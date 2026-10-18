# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""deep_v2 general search tier, decided after stance (M147, use_cases/retrieval/tiered_search.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.use_cases.retrieval.tiered_search import clusters_insufficient_after_stance, merge_round
from spectrue_core.utils.trace import Trace


@dataclass
class GeneralTierSearchStep:
    """Second retrieval round (general domains) only for clusters still insufficient after stance."""

    round_steps: tuple[Any, ...]  # search(tier="general") -> attribution -> enrich -> collect -> stance steps
    name: str = "general_tier_search"
    weight: float = 30.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        plans = ctx.get_extra("cluster_search_plans", []) or []
        insufficient = clusters_insufficient_after_stance(plans, ctx.sources or [])
        Trace.event("retrieval.general_tier.plan", {"clusters_total": len(plans), "clusters_insufficient": sorted(insufficient)})
        if not insufficient:
            return ctx
        sub = (ctx.set_extra("cluster_search_plans", [p for p in plans if str(p.get("cluster_id") or "cluster") in insufficient])
               .set_extra("stance_annotations", None))
        for step in self.round_steps:
            sub = await step.run(sub)
        return merge_round(ctx, sub)
