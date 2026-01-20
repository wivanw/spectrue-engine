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

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResultAssemblyStep:
    """
    Assemble result for extraction-only mode.
    
    Returns specific payload with extracted claims and prepared text.
    """
    name = "extraction_result_assembly"
    weight: float = 1.0
    
    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Assemble result for extraction-only mode."""
        try:
            # Get ledger/metering for cost calculation
            ledger = ctx.get_extra("ledger")
            cost_summary = ledger.to_summary_dict() if ledger else None
            
            # Use prepared_fact (from PrepareInputStep) or fallback to raw
            fact = ctx.get_extra("prepared_fact", ctx.get_extra("raw_fact", ""))
            
            final_result = {
                 "status": "ok",
                 "verified_score": 0.0,
                 "text": fact,
                 "sources": ctx.sources or [],
                 "details": [],
                 "_extracted_claims": ctx.claims or [],
                 "cost_summary": cost_summary,
                 # Legacy cost field (optional, computed from summary usually)
                 "cost": cost_summary.get("total_tokens_cost", 0.0) if cost_summary else 0.0,
            }
            
            return ctx.set_extra("final_result", final_result)
            
        except Exception as e:
            logger.exception("[ExtractionResultAssemblyStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
