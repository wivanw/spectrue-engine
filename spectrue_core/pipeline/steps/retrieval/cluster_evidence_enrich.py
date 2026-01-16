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
from spectrue_core.utils.trace import Trace


def _claim_text_map(claims: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx, claim in enumerate(claims or []):
        if not isinstance(claim, dict):
            continue
        cid = claim.get("id") or claim.get("claim_id") or f"c{idx + 1}"
        text = claim.get("normalized_text") or claim.get("text") or ""
        mapping[str(cid)] = text
    return mapping


@dataclass
class ClusterEvidenceEnrichStep:
    """Run EAL on per-claim evidence produced by cluster attribution."""

    config: Any
    search_mgr: Any
    name: str = "cluster_evidence_enrich"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            evidence_by_claim = ctx.get_extra("evidence_by_claim")
            if not isinstance(evidence_by_claim, dict) or not evidence_by_claim:
                Trace.event("cluster_evidence_enrich.skipped", {"reason": "no_evidence_by_claim"})
                return ctx

            claim_texts = _claim_text_map(ctx.claims or [])

            enriched_by_claim: dict[str, list[dict[str, Any]]] = {}
            enriched_sources: list[dict[str, Any]] = []

            for claim_id, items in evidence_by_claim.items():
                if not isinstance(items, (list, tuple)):
                    continue
                claim_id_str = str(claim_id)
                claim_text = claim_texts.get(claim_id_str, "")
                sources: list[dict[str, Any]] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    src = dict(item)
                    if claim_text:
                        src.setdefault("claim_text", claim_text)
                    if src.get("relevance_score") is None and src.get("similarity_score") is not None:
                        src["relevance_score"] = src.get("similarity_score")
                    sources.append(src)
                if not sources:
                    continue

                enriched = await self.search_mgr.apply_evidence_acquisition_ladder(
                    sources,
                    budget_context="claim",
                    claim_id=claim_id_str,
                    cache_only=True,
                )
                enriched_by_claim[claim_id_str] = enriched
                enriched_sources.extend(enriched)

            Trace.event(
                "cluster_evidence_enrich.completed",
                {
                    "claims": len(enriched_by_claim),
                    "sources": len(enriched_sources),
                },
            )

            return (
                ctx.with_update(sources=enriched_sources)
                .set_extra("evidence_by_claim", enriched_by_claim)
            )
        except Exception as exc:
            raise PipelineExecutionError(self.name, str(exc), cause=exc) from exc
