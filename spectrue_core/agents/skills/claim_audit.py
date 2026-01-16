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
"""Claim audit skill for structured audit annotations."""

from __future__ import annotations

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import CLAIM_AUDIT_SCHEMA
from spectrue_core.agents.skills.audit_prompts import (
    build_claim_audit_prompt,
    build_claim_audit_system_prompt,
)
from spectrue_core.schema.claim_frame import ClaimFrame
from spectrue_core.schema.rgba_audit import ClaimAudit
from spectrue_core.utils.trace import Trace


class ClaimAuditSkill:
    """LLM-driven claim audit annotation (no scoring)."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def audit(self, frame: ClaimFrame) -> ClaimAudit:
        user_prompt = build_claim_audit_prompt(frame)
        system_prompt = build_claim_audit_system_prompt()

        Trace.event("claim_audit.start", {"claim_id": frame.claim_id})

        try:
            response = await self.llm.call_structured(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CLAIM_AUDIT_SCHEMA,
                schema_name="claim_audit",
                model="gpt-5-nano",
                temperature=0,
            )
            audit = ClaimAudit(**response)
            if audit.claim_id != frame.claim_id:
                raise ValueError("claim_id mismatch in audit response")
            Trace.event("claim_audit.complete", {"claim_id": frame.claim_id})
            return audit
        except Exception as exc:
            Trace.event("claim_audit.error", {"claim_id": frame.claim_id, "error": str(exc)})
            raise
