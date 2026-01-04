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

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFlowStep:
    """
    Deprecated alias for EvidenceCollectStep (collection only).

    This step no longer performs judging. Configure `enable_global_scoring`
    to control whether a global evidence pack is produced.
    """

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    enable_global_scoring: bool = True
    name: str = "evidence_flow"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            from spectrue_core.pipeline.steps.evidence_collect import EvidenceCollectStep

            if not self.enable_global_scoring:
                Trace.event(
                    "evidence_flow.skip_global_scoring",
                    {"reason": "collection_only", "claims_count": len(ctx.claims)},
                )

            collector = EvidenceCollectStep(
                agent=self.agent,
                search_mgr=self.search_mgr,
                cluster_evidence=self.enable_global_scoring,
                include_global_pack=self.enable_global_scoring,
            )

            return await collector.run(ctx)

        except Exception as e:
            logger.exception("[EvidenceFlowStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
