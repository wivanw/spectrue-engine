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

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.skills.evidence_audit import EvidenceAuditSkill
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext
from spectrue_core.schema.rgba_audit import EvidenceAudit, RGBAStatus
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class AuditEvidenceStep:
    """Produce evidence-level audit records via LLM."""

    llm_client: LLMClient
    name: str = "audit_evidence"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            Trace.phase_start("audit_evidence")
            deep_ctx: DeepClaimContext = ctx.get_extra("deep_claim_ctx", DeepClaimContext())

            if not deep_ctx.claim_frames:
                Trace.event("evidence_audit.skip", {"reason": "no_frames"})
                return ctx

            skill = EvidenceAuditSkill(self.llm_client)
            errors: dict[str, Any] = dict(ctx.get_extra("audit_errors") or {})
            evidence_errors = dict(errors.get("evidence_audit", {}))
            audits: list[EvidenceAudit] = []

            tasks = []
            for frame in deep_ctx.claim_frames:
                for evidence in frame.evidence_items:
                    tasks.append((frame, evidence))

            if not tasks:
                Trace.event("evidence_audit.skip", {"reason": "no_evidence"})
                return ctx

            async def audit_one(frame, evidence):
                try:
                    audit = await skill.audit(frame, evidence)
                    return ("ok", evidence.evidence_id, audit)
                except Exception as exc:
                    return ("error", evidence.evidence_id, exc)

            results = await asyncio.gather(
                *[audit_one(frame, evidence) for frame, evidence in tasks],
                return_exceptions=False,
            )

            for status, evidence_id, payload in results:
                if status == "ok":
                    audits.append(payload)
                else:
                    evidence_errors[str(evidence_id)] = {
                        "status": RGBAStatus.PIPELINE_ERROR,
                        "error_type": "audit_failed",
                        "message": str(payload),
                    }

            if evidence_errors:
                errors["evidence_audit"] = evidence_errors

            Trace.event(
                "evidence_audit.complete",
                {
                    "count": len(audits),
                    "error_count": len(evidence_errors),
                },
            )

            return (
                ctx.set_extra("evidence_audits", audits)
                .set_extra("audit_errors", errors)
            )

        except Exception as e:
            logger.exception("[AuditEvidenceStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
        finally:
            Trace.phase_end("audit_evidence")
