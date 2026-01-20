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

from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.cluster_attribution import (
    attribute_cluster_evidence,
)
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url


@dataclass
class ClusterAttributionStep:
    """Attribute cluster evidence to claims with precision/corroboration tags."""

    config: Any
    name: str = "cluster_attribution"
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            claims = [c for c in (ctx.claims or []) if isinstance(c, dict)]
            cluster_claims = ctx.get_extra("cluster_claims", {}) or {}
            cluster_docs = ctx.get_extra("cluster_evidence_docs", {}) or {}

            if not claims or not cluster_claims or not cluster_docs:
                Trace.event("retrieval.cluster_attribution.skipped", {"reason": "missing_inputs"})
                return ctx

            runtime = getattr(self.config, "runtime", None)
            deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())

            result = await attribute_cluster_evidence(
                claims=claims,
                cluster_claims=cluster_claims,
                cluster_evidence_docs=cluster_docs,
                deep_v2_cfg=deep_v2_cfg,
            )

            Trace.event(
                "retrieval.cluster_attribution.completed",
                {
                    "claims": len(result.evidence_by_claim),
                    "sources": len(result.sources),
                },
            )

            evidence_item_meta: dict[str, dict[str, Any]] = {}
            for item in result.sources:
                if not isinstance(item, dict):
                    continue
                claim_id = item.get("claim_id")
                url = item.get("url") or item.get("link")
                if not claim_id or not url:
                    continue
                canonical = normalize_url(str(url)) or str(url)
                key = f"{claim_id}::{canonical}"
                evidence_item_meta[key] = {
                    "content_hash": item.get("content_hash"),
                    "publisher_id": item.get("publisher_id"),
                    "similar_cluster_id": item.get("similar_cluster_id"),
                    "attribution": item.get("attribution"),
                }

            return (
                ctx.with_update(sources=result.sources)
                .set_extra("evidence_by_claim", result.evidence_by_claim)
                .set_extra("evidence_item_meta", evidence_item_meta)
                .set_extra("cluster_attribution", {
                    "claims": len(result.evidence_by_claim),
                    "sources": len(result.sources),
                })
            )
        except Exception as exc:
            raise PipelineExecutionError(self.name, str(exc), cause=exc) from exc
