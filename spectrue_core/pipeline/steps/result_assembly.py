# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from __future__ import annotations

import logging
from dataclasses import dataclass

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ResultAssemblyStep:
    """Assemble final result payload."""

    name: str = "result_assembly"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Assemble final result."""
        try:
            verdict = ctx.verdict or {}
            sources = ctx.sources or []
            ledger = ctx.get_extra("ledger")

            cost_summary = ledger.to_summary_dict() if ledger else None

            final_result = {
                "status": "ok",
                "verified_score": verdict.get("verified_score", 0.0),
                "explainability_score": verdict.get("explainability_score", 0.0),
                "danger_score": verdict.get("danger_score", 0.0),
                "style_score": verdict.get("style_score", 0.0),
                "bias_score": verdict.get("style_score", 0.0), # Compat alias
                "rgba": ctx.get_extra("rgba", [0.0, 0.0, 0.0, 0.5]),
                "sources": sources,
                "rationale": verdict.get("rationale", ""),
                "analysis": verdict.get("analysis") or verdict.get("rationale", ""), # Legacy compat
                "cost_summary": cost_summary,
                "cost": verdict.get("cost", 0.0),
            }

            if not ledger:
                logger.warning(f"[ResultAssemblyStep] Ledger not found in context. Cost summary will be empty/zero. Extras keys: {list(ctx.extras.keys())}")

            if ctx.get_extra("oracle_hit"):
                final_result["oracle_hit"] = True

            Trace.event("result_assembly.completed", {"verified_score": final_result["verified_score"]})

            return ctx.set_extra("final_result", final_result)


        except Exception as e:
            logger.exception("[ResultAssemblyStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
