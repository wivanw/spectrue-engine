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

from __future__ import annotations

import logging
from dataclasses import dataclass

from spectrue_core.use_cases.claims.clustering import build_soft_clusters
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ClaimClusterStep(Step):
    """
    Build soft clusters of related claims using the claim graph.
    This step does NOT merge claims and does NOT affect scoring.
    It only produces metadata for downstream reconciliation.
    """

    name: str = "claim_cluster"
    weight: float = 5.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        graph_result = ctx.get_extra("graph_result")
        if not graph_result:
            Trace.event("claim_cluster.skipped", {"reason": "no_claim_graph"})
            return ctx

        clusters, cluster_map = build_soft_clusters(
            claims=ctx.claims,
            graph_result=graph_result,
        )

        Trace.event("claim_cluster.completed", {
            "cluster_count": len(clusters),
            "avg_cluster_size": (
                sum(len(c["claim_ids"]) for c in clusters) / len(clusters)
                if clusters else 0
            ),
        })

        return (
            ctx.set_extra("claim_clusters", clusters)
            .set_extra("cluster_map", cluster_map)
        )
