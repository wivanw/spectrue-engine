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
"""Evidence audit skill for structured audit annotations."""

from __future__ import annotations

import re

from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import EVIDENCE_AUDIT_SCHEMA
from spectrue_core.agents.skills.audit_prompts import (
    build_evidence_audit_prompt,
    build_evidence_audit_system_prompt,
)
from spectrue_core.schema.claim_frame import ClaimFrame, EvidenceItemFrame
from spectrue_core.schema.rgba_audit import EvidenceAudit
from spectrue_core.utils.trace import Trace
from spectrue_core.llm.model_registry import ModelID


class EvidenceAuditSkill:
    """LLM-driven evidence audit annotation (no scoring)."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def audit(self, frame: ClaimFrame, evidence: EvidenceItemFrame) -> EvidenceAudit:
        user_prompt = build_evidence_audit_prompt(frame, evidence)
        system_prompt = build_evidence_audit_system_prompt()
        source_id = evidence.source_id or evidence.url

        def _is_index_source(value: str) -> bool:
            if not value:
                return False
            return bool(re.match(r"^(source_)?\d+$", value) or re.match(r"^s\d+$", value))

        async def _call_audit(prompt: str) -> EvidenceAudit:
            response = await self.llm.call_structured(
                user_prompt=user_prompt,
                system_prompt=prompt,
                schema=EVIDENCE_AUDIT_SCHEMA,
                schema_name="evidence_audit",
                model=ModelID.NANO,
                temperature=0,
            )
            return EvidenceAudit(**response)

        Trace.event(
            "evidence_audit.start",
            {"claim_id": frame.claim_id, "evidence_id": evidence.evidence_id},
        )

        try:
            audit = await _call_audit(system_prompt)
            if audit.claim_id != frame.claim_id:
                raise ValueError("claim_id mismatch in audit response")
            if audit.evidence_id != evidence.evidence_id:
                raise ValueError("evidence_id mismatch in audit response")
            if audit.source_id != source_id:
                if _is_index_source(audit.source_id):
                    Trace.event(
                        "evidence_audit.remap_source_id",
                        {
                            "claim_id": frame.claim_id,
                            "evidence_id": evidence.evidence_id,
                            "received": audit.source_id,
                        },
                    )
                    audit = audit.model_copy(update={"source_id": source_id})
                else:
                    Trace.event(
                        "evidence_audit.retry",
                        {
                            "claim_id": frame.claim_id,
                            "evidence_id": evidence.evidence_id,
                            "reason": "source_id_mismatch",
                            "received": audit.source_id,
                        },
                    )
                    constrained = (
                        f"{system_prompt}\n"
                        f"Return source_id exactly: {source_id}\n"
                    )
                    audit = await _call_audit(constrained)
                    if audit.source_id != source_id:
                        raise ValueError("source_id mismatch in audit response")
            Trace.event(
                "evidence_audit.complete",
                {"claim_id": frame.claim_id, "evidence_id": evidence.evidence_id},
            )
            return audit
        except Exception as exc:
            Trace.event(
                "evidence_audit.error",
                {"claim_id": frame.claim_id, "evidence_id": evidence.evidence_id, "error": str(exc)},
            )
            raise
