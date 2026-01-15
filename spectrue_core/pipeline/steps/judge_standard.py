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

from spectrue_core.pipeline.contracts import JUDGMENTS_KEY, Judgments
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.verification.pipeline.pipeline_evidence import (
    EvidenceFlowInput,
    score_evidence_collection,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgeStandardStep:
    """Run the standard (single-call) judge on collected evidence."""

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    name: str = "judge_standard"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            if ctx.get_extra("gating_rejected") or ctx.get_extra("oracle_hit"):
                return ctx

            if ctx.verdict:
                return ctx

            collection = ctx.get_extra("evidence_collection")
            if collection is None:
                raise PipelineExecutionError(
                    self.name, "Missing evidence_collection in context"
                )

            inp = EvidenceFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                original_fact=ctx.get_extra("original_fact", ""),
                lang=ctx.lang,
                content_lang=ctx.lang,
                analysis_mode=ctx.mode.api_analysis_mode,
                progress_callback=ctx.get_extra("progress_callback"),
                prior_belief=ctx.get_extra("prior_belief"),
                context_graph=ctx.get_extra("context_graph"),
                claim_extraction_text=ctx.get_extra("claim_extraction_text", ""),
            )

            Trace.event(
                "judge_standard.invoked",
                {
                    "claim_count": len(collection.claims),
                    "sources_count": len(collection.sources),
                },
            )

            result = await score_evidence_collection(
                agent=self.agent,
                search_mgr=self.search_mgr,
                enrich_sources_with_trust=enrich_sources_with_trust,
                inp=inp,
                collection=collection,
                score_mode="standard",
            )

            Trace.event(
                "standard.article_judged",
                {
                    "count": 1,
                    "claim_count": len(collection.claims),
                },
            )

            Trace.event(
                "judge_standard.completed",
                {"verified_score": result.get("verified_score")},
            )

            judgments = Judgments(standard=result, deep=tuple())

            return (
                ctx.with_update(verdict=result)
                .set_extra("rgba", result.get("rgba"))
                .set_extra(JUDGMENTS_KEY, judgments)
            )

        except Exception as e:
            logger.exception("[JudgeStandardStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
