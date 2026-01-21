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
from spectrue_core.agents.skills.claim_audit import ClaimAuditSkill
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.pipeline.steps.deep_claim import DeepClaimContext
from spectrue_core.schema.rgba_audit import ClaimAudit, RGBAStatus
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class AuditClaimsStep:
    """Produce claim-level audit records via LLM."""

    llm_client: LLMClient
    name: str = "audit_claims"
    weight: float = 15.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            Trace.phase_start("audit_claims")
            deep_ctx: DeepClaimContext = ctx.get_extra("deep_claim_ctx", DeepClaimContext())

            if not deep_ctx.claim_frames:
                Trace.event("claim_audit.skip", {"reason": "no_frames"})
                return ctx

            skill = ClaimAuditSkill(self.llm_client)
            errors: dict[str, Any] = dict(ctx.get_extra("audit_errors") or {})
            claim_errors = dict(errors.get("claim_audit", {}))
            audits: list[ClaimAudit] = []

            async def audit_one(frame):
                try:
                    audit = await skill.audit(frame)
                    return ("ok", frame.claim_id, audit)
                except Exception as exc:
                    return ("error", frame.claim_id, exc)

            results = await asyncio.gather(
                *[audit_one(frame) for frame in deep_ctx.claim_frames],
                return_exceptions=False,
            )

            for status, claim_id, payload in results:
                if status == "ok":
                    audits.append(payload)
                else:
                    claim_errors[str(claim_id)] = {
                        "status": RGBAStatus.PIPELINE_ERROR,
                        "error_type": "audit_failed",
                        "message": str(payload),
                    }

            if claim_errors:
                errors["claim_audit"] = claim_errors

            Trace.event(
                "claim_audit.complete",
                {
                    "count": len(audits),
                    "error_count": len(claim_errors),
                },
            )

            return (
                ctx.set_extra("claim_audits", audits)
                .set_extra("audit_errors", errors)
            )

        except Exception as e:
            logger.exception("[AuditClaimsStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
        finally:
            Trace.phase_end("audit_claims")
