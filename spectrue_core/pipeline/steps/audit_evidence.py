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

from spectrue_core.llm.llm_client import LLMClient
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext
from spectrue_core.schema.rgba_audit import EvidenceAudit, RGBAStatus
from spectrue_core.use_cases.evidence.audit import run_audit
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class AuditEvidenceStep:
    """Produce evidence-level audit records via LLM."""

    llm_client: LLMClient
    name: str = "audit_evidence"
    weight: float = 17.0  # ~17s actual

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            Trace.phase_start("audit_evidence")
            deep_ctx: DeepClaimContext = ctx.get_extra("deep_claim_ctx", DeepClaimContext())

            if not deep_ctx.claim_frames:
                Trace.event("evidence_audit.skip", {"reason": "no_frames"})
                return ctx

            errors: dict[str, Any] = dict(ctx.get_extra("audit_errors") or {})
            evidence_errors = dict(errors.get("evidence_audit", {}))

            if not deep_ctx.claim_frames:
                Trace.event("evidence_audit.skip", {"reason": "no_evidence"})
                return ctx
            result = await run_audit(
                claim_frames=deep_ctx.claim_frames,
                llm_client=self.llm_client,
                error_status=RGBAStatus.PIPELINE_ERROR,
            )

            audits: list[EvidenceAudit] = list(result.audits)
            evidence_errors.update(result.errors)

            if evidence_errors:
                errors["evidence_audit"] = evidence_errors

            Trace.event(
                "evidence_audit.complete",
                {
                    "count": len(audits),
                    "error_count": len(evidence_errors),
                },
            )

            return ctx.set_extra("evidence_audits", audits).set_extra("audit_errors", errors)

        except Exception as e:
            logger.exception("[AuditEvidenceStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
        finally:
            Trace.phase_end("audit_evidence")
