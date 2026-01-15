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
from typing import Any

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

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        graph_result = ctx.get_extra("graph_result")
        if not graph_result:
            Trace.event("claim_cluster.skipped", {"reason": "no_claim_graph"})
            return ctx

        # Map claim IDs to their objects (dicts) from the context
        claim_map = {str(c.get("id")): c for c in ctx.claims if c.get("id")}
        
        clusters = []
        # connected_components() returns list of lists of claim IDs
        for component_ids in graph_result.connected_components():
            component_claims = [claim_map[cid] for cid in component_ids if cid in claim_map]
            if not component_claims:
                continue

            shared_assertions = set()
            topic_tags = set()

            for claim in component_claims:
                # 1. Collect assertion keys (non-heuristic metadata)
                assertions = claim.get("assertions", [])
                if isinstance(assertions, list):
                    for a in assertions:
                        key = None
                        if hasattr(a, "key"):
                            key = a.key
                        elif isinstance(a, dict):
                            key = a.get("key")
                        if key:
                            shared_assertions.add(key)
                
                # 2. Collect topic tags (from ClaimMetadata object)
                metadata = claim.get("metadata")
                if metadata:
                    # Attempt to get topic_tags from datatlass or dict
                    tags = getattr(metadata, "topic_tags", []) if not isinstance(metadata, dict) else metadata.get("topic_tags", [])
                    if isinstance(tags, (list, set)):
                        topic_tags.update(tags)

            clusters.append({
                "cluster_id": f"cluster_{len(clusters)}",
                "claim_ids": [str(c.get("id")) for c in component_claims],
                "shared_assertion_keys": list(shared_assertions),
                "topic_tags": list(topic_tags),
            })

        Trace.event("claim_cluster.completed", {
            "cluster_count": len(clusters),
            "avg_cluster_size": (
                sum(len(c["claim_ids"]) for c in clusters) / len(clusters)
                if clusters else 0
            ),
        })

        return ctx.set_extra("claim_clusters", clusters)
