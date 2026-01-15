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
from typing import Any

from spectrue_core.graph.claim_clusters import build_claim_clusters
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.utils.trace import Trace


@dataclass
class ClaimClustersStep(Step):
    config: Any

    @property
    def name(self) -> str:
        return "claim_clusters"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        claims = ctx.claims or []
        if not claims:
            return (
                ctx.set_extra("claim_clusters", [])
                .set_extra("clusters_summary", [])
                .set_extra("cluster_claims", {})
                .set_extra("cluster_representatives", {})
                .set_extra("cluster_map", {})
            )

        graph_result = ctx.get_extra("graph_result")
        runtime = getattr(self.config, "runtime", None)
        deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())

        clusters = build_claim_clusters(
            claims=claims,
            graph_result=graph_result,
            quantile=deep_v2_cfg.claim_cluster_quantile,
            representative_min_k=deep_v2_cfg.representative_min_k,
            representative_max_k=deep_v2_cfg.representative_max_k,
        )

        cluster_map: dict[str, str] = {}
        cluster_claims: dict[str, list[dict[str, Any]]] = {}
        representative_claims: dict[str, list[dict[str, Any]]] = {}

        claim_lookup: dict[str, dict[str, Any]] = {}
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("id") or claim.get("claim_id") or f"c{idx + 1}")
            # Create a copy to avoid mutating original claims (immutability contract)
            claim_lookup[claim_id] = dict(claim)

        for cluster in clusters:
            cluster_claims[cluster.cluster_id] = []
            representative_claims[cluster.cluster_id] = []
            for claim_id in cluster.claim_ids:
                claim = claim_lookup.get(claim_id)
                if claim is None:
                    continue
                # Modify the copy, not the original
                claim["cluster_id"] = cluster.cluster_id
                cluster_map[claim_id] = cluster.cluster_id
                cluster_claims[cluster.cluster_id].append(claim)
            for claim_id in cluster.representative_claim_ids:
                claim = claim_lookup.get(claim_id)
                if claim is not None:
                    representative_claims[cluster.cluster_id].append(claim)

        Trace.event(
            "claim_clusters.built",
            {
                "clusters": len(clusters),
                "cluster_sizes": [c.size for c in clusters],
            },
        )

        return (
            ctx.set_extra("claim_clusters", clusters)
            .set_extra("clusters_summary", [c.to_summary() for c in clusters])
            .set_extra("cluster_claims", cluster_claims)
            .set_extra("cluster_representatives", representative_claims)
            .set_extra("cluster_map", cluster_map)
        )
