# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""Post-search relevance gate G1 (M145 Phase R; restores M66 semantic gating).

Runs between web search and rerank. Writes the Bayesian posterior similarity of
each per-claim result back into the raw payload, so `RerankStep` ranks and caps
with it — the single ranking implementation decides what is kept.
Logic: `use_cases.retrieval.decision_gates.fuse_result_similarity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from spectrue_core.pipeline.contracts import RAW_SEARCH_RESULTS_KEY, RawSearchResults
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors
from spectrue_core.use_cases.retrieval.decision_gates import claims_by_id, fuse_result_similarity


@dataclass
class SearchRelevanceGateStep:
    agent: Any
    decisions: DecisionsConfig = field(default_factory=DecisionsConfig)  # all gates off
    sensors: dict[str, DecisionSensorCalibration] = field(default_factory=uncalibrated_sensors)
    name: str = "search_relevance_gate"
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        raw: RawSearchResults | None = ctx.get_extra(RAW_SEARCH_RESULTS_KEY)
        if not raw or not raw.results:
            return ctx
        claims = claims_by_id(list(ctx.claims or []) + list(ctx.get_extra("eligible_claims", []) or []))
        entries = [(r.claim_id, r.provider_payload) for r in raw.results]
        sims = await fuse_result_similarity(self.agent, self.decisions, self.sensors, entries, claims, ctx.lang or "en")
        if sims is None:
            return ctx
        results = tuple(
            replace(r, provider_payload=[
                {**src, "similarity_score": sims[(r.claim_id, str(src.get("url") or src.get("link")))]}
                if (r.claim_id, str(src.get("url") or src.get("link"))) in sims else src
                for src in r.provider_payload
            ])
            for r in raw.results
        )
        return ctx.set_extra(RAW_SEARCH_RESULTS_KEY, replace(raw, results=results))
