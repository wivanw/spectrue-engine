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
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace, current_trace_id

logger = logging.getLogger(__name__)


@dataclass
class MeteringSetupStep:
    """
    Initialize cost metering infrastructure.

    Sets up CostLedger, TavilyMeter, LLMMeter, and EmbedService metering.
    This must run before any billable operations.

    Context Input:
        - config (via extras)

    Context Output:
        - ledger (via extras)
        - tavily_meter (via extras)
        - llm_meter (via extras)
    """

    config: Any  # SpectrueConfig
    name: str = "metering_setup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Initialize metering infrastructure."""
        from spectrue_core.billing.cost_ledger import CostLedger
        from spectrue_core.billing.metering import TavilyMeter, LLMMeter
        from spectrue_core.billing.config_loader import load_pricing_policy
        from spectrue_core.billing.progress_emitter import CostProgressEmitter
        from spectrue_core.utils.embedding_service import EmbedService
        from spectrue_core.verification.pipeline.pipeline_metering import create_progress_callback

        try:
            policy = load_pricing_policy()
            ledger = CostLedger(run_id=current_trace_id())
            tavily_meter = TavilyMeter(ledger=ledger, policy=policy)
            llm_meter = LLMMeter(ledger=ledger, policy=policy)

            # Configure embedding metering
            EmbedService.configure(
                openai_api_key=getattr(self.config, "openai_api_key", None),
                meter=llm_meter,
                meter_stage="embed",
            )

            progress_emitter = CostProgressEmitter(
                ledger=ledger,
                min_delta_to_show=policy.min_delta_to_show,
                emit_cost_deltas=policy.emit_cost_deltas,
            )
            
            # Wrap progress callback to include costs
            raw_cb = ctx.get_extra("progress_callback")
            if raw_cb:
                # Use helper (imported)
                wrapped_cb = await create_progress_callback(raw_cb, progress_emitter)
            else:
                wrapped_cb = None

            Trace.event(
                "metering_setup.completed",
                {"run_id": current_trace_id()},
            )

            return (
                ctx.set_extra("ledger", ledger)
                .set_extra("tavily_meter", tavily_meter)
                .set_extra("llm_meter", llm_meter)
                .set_extra("progress_emitter", progress_emitter)
                .set_extra("pricing_policy", policy)
                .set_extra("progress_callback", wrapped_cb)
            )

        except Exception as e:
            logger.exception("[MeteringSetupStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
