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

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.pipeline.pipeline_evidence import (
    EvidenceFlowInput,
    annotate_evidence_stance,
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
class StanceAnnotateStep:
    """Optional stance annotation for collected sources."""

    agent: Any  # FactCheckerAgent
    name: str = "stance_annotate"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            if ctx.get_extra("gating_rejected") or ctx.get_extra("oracle_hit"):
                return ctx

            if not ctx.claims or not ctx.sources:
                return ctx

            inp = EvidenceFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                original_fact=ctx.get_extra("original_fact", ""),
                lang=ctx.lang,
                content_lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                progress_callback=ctx.get_extra("progress_callback"),
            )

            annotated = await annotate_evidence_stance(
                agent=self.agent,
                inp=inp,
                claims=ctx.claims,
                sources=ctx.sources,
            )

            if annotated:
                Trace.event(
                    "stance_annotate.completed",
                    {"count": len(annotated)},
                )
                return (
                    ctx.with_update(sources=annotated)
                    .set_extra("stance_annotations", annotated)
                    .set_extra("evidence_by_claim", _group_sources_by_claim(annotated))
                )

            return ctx

        except Exception as e:
            logger.exception("[StanceAnnotateStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
