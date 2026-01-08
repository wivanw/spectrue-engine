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

from spectrue_core.pipeline.contracts import RGBA_AUDIT_KEY
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.scoring.rgba_audit.aggregation import aggregate_rgba_audit

logger = logging.getLogger(__name__)


@dataclass
class AggregateRGBAAuditStep:
    """Aggregate claim and evidence audits into RGBAResult."""

    name: str = "aggregate_rgba_audit"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            claim_audits = ctx.get_extra("claim_audits") or []
            evidence_audits = ctx.get_extra("evidence_audits") or []
            audit_sources = ctx.get_extra("audit_sources") or []
            trace_context = ctx.get_extra("audit_trace_context") or ctx.get_extra("trace_context") or {}
            audit_errors = ctx.get_extra("audit_errors") or {}

            rgba_result = aggregate_rgba_audit(
                claim_audits=claim_audits,
                evidence_audits=evidence_audits,
                sources=audit_sources,
                trace_context=trace_context,
                audit_errors=audit_errors,
            )

            Trace.event(
                "rgba_audit.aggregate.complete",
                {
                    "status": {
                        "R": rgba_result.R.status.value,
                        "G": rgba_result.G.status.value,
                        "B": rgba_result.B.status.value,
                        "A": rgba_result.A.status.value,
                    },
                },
            )

            return ctx.set_extra(RGBA_AUDIT_KEY, rgba_result)
        except Exception as e:
            logger.exception("[AggregateRGBAAuditStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
