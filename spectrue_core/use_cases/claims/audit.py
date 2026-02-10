"""Claim audit use cases."""

from __future__ import annotations

import asyncio
from typing import Any

from spectrue_core.adapters.llm.claim_audit import ClaimAuditSkill
from spectrue_core.domain.verification.verdict.model import RGBAStatus


async def run_claim_audit(*, claim_frames: list[Any], llm_client: Any):
    skill = ClaimAuditSkill(llm_client)
    audits = []
    claim_errors: dict[str, dict[str, Any]] = {}

    async def audit_one(frame):
        try:
            audit = await skill.audit(frame)
            return ("ok", frame.claim_id, audit)
        except Exception as exc:
            return ("error", frame.claim_id, exc)

    results = await asyncio.gather(
        *[audit_one(frame) for frame in claim_frames],
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

    return audits, claim_errors
