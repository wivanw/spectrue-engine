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
from decimal import Decimal
import math

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class CostSummaryStep:
    """Attach metering summary to the final result payload."""

    name: str = "cost_summary"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ledger = ctx.get_extra("ledger")
            if ledger is None:
                Trace.event("cost_summary.missing_ledger", {})
                return ctx.set_extra("cost_summary", None)

            summary = ledger.to_summary_dict()
            credits_used = summary.get("credits_used")
            if credits_used is None:
                total_credits = summary.get("total_credits")
                if isinstance(total_credits, (int, float, Decimal)):
                    credits_used = int(math.ceil(float(total_credits)))
                    summary["credits_used"] = credits_used
            final_result = ctx.get_extra("final_result")
            if isinstance(final_result, dict):
                updated = dict(final_result)
                updated["cost_summary"] = summary
                if credits_used is not None and "credits" not in updated:
                    updated["credits"] = credits_used
                ctx = ctx.set_extra("final_result", updated)

            Trace.event(
                "cost_summary.attached",
                {"event_count": len(summary.get("events", []))},
            )

            return ctx.set_extra("cost_summary", summary)

        except Exception as e:
            logger.exception("[CostSummaryStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
