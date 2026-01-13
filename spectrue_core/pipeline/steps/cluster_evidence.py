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

from spectrue_core.pipeline.contracts import GATES_KEY, Gates
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.pipeline.pipeline_evidence import (
    EvidenceFlowInput,
    annotate_evidence_stance,
    rebuild_evidence_pack,
)

logger = logging.getLogger(__name__)


def _group_sources_by_claim(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        claim_id = source.get("claim_id")
        if not claim_id:
            continue
        grouped.setdefault(str(claim_id), []).append(source)
    return grouped


@dataclass
class ClusterEvidenceStep:
    """Optional clustering to rebuild evidence pack with stances.

    Controlled by EVOI gating policy. Reads ctx.extras["gates"].
    """

    agent: Any  # FactCheckerAgent
    name: str = "cluster_evidence"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            if ctx.get_extra("gating_rejected") or ctx.get_extra("oracle_hit"):
                return ctx

            # Check gate policy (EVOI decision from SemanticGatingStep)
            gates: Gates | None = ctx.get_extra(GATES_KEY)
            if gates is not None and not gates.is_cluster_enabled():
                Trace.event(
                    "cluster_evidence.skipped_by_gate",
                    {
                        "reasons": list(gates.cluster.reasons) if gates.cluster else [],
                        "p_need": gates.cluster.p_need if gates.cluster else 0,
                    },
                )
                return ctx

            collection = ctx.get_extra("evidence_collection")
            if collection is None:
                return ctx

            inp = EvidenceFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                original_fact=ctx.get_extra("original_fact", ""),
                lang=ctx.lang,
                content_lang=ctx.lang,
                search_type=ctx.search_type,
                progress_callback=ctx.get_extra("progress_callback"),
            )

            clustered = ctx.get_extra("stance_annotations")
            if not clustered:
                clustered = await annotate_evidence_stance(
                    agent=self.agent,
                    inp=inp,
                    claims=collection.claims,
                    sources=collection.sources,
                )

            if not clustered:
                return ctx

            from spectrue_core.verification.evidence.evidence import build_evidence_pack

            updated_collection = rebuild_evidence_pack(
                build_evidence_pack=build_evidence_pack,
                collection=collection,
                clustered_results=clustered,
                sources_override=clustered,
                inp=inp,
            )

            Trace.event(
                "cluster_evidence.completed",
                {"count": len(clustered)},
            )

            return (
                ctx.with_update(sources=clustered, evidence=updated_collection.pack)
                .set_extra("evidence_collection", updated_collection)
                .set_extra("evidence_by_claim", _group_sources_by_claim(clustered))
                .set_extra("stance_annotations", clustered)
            )

        except Exception as e:
            logger.exception("[ClusterEvidenceStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
