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

import asyncio
import re
from typing import Any

from spectrue_core.llm.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import EVIDENCE_AUDIT_SCHEMA
from spectrue_core.adapters.llm.audit_prompts import (
    build_evidence_audit_prompt,
    build_evidence_audit_system_prompt,
)
from spectrue_core.domain.claims.frame import ClaimFrame, EvidenceItemFrame
from spectrue_core.domain.verification.verdict.model import EvidenceAudit
from spectrue_core.utils.trace import Trace
from spectrue_core.llm.model_registry import ModelID


class EvidenceAuditSkill:
    """LLM-driven evidence audit annotation (no scoring)."""

    def __init__(self, llm_client: LLMClient, decisions: Any):
        self.llm = llm_client
        self.decisions = decisions

    async def audit(self, frame: ClaimFrame, evidence: EvidenceItemFrame) -> EvidenceAudit:
        """M147 T021 gate "evidence_audit": off = LLM audit; shadow = LLM audit + sensor compare;
        on = the audit is built from the Decisions sensor (no LLM call); a missing reading falls back to the LLM."""
        from spectrue_core.adapters.llm.decision_gates import gate_mode
        from spectrue_core.use_cases.evidence.audit_sensor import audit_from_reading, compare_audit

        mode = gate_mode(self.decisions, "evidence_audit")
        if mode == "off":
            return await self._audit_llm(frame, evidence)
        from spectrue_core.adapters.llm.evidence_audit_sensor import read_evidence_audit

        lang = getattr(frame, "claim_language", None) or "en"
        if mode == "on":
            reading = await read_evidence_audit(self.llm, self.decisions, frame, evidence, lang)
            if reading is not None:
                return audit_from_reading(frame, evidence, reading)
            return await self._audit_llm(frame, evidence)
        audit, reading = await asyncio.gather(
            self._audit_llm(frame, evidence), read_evidence_audit(self.llm, self.decisions, frame, evidence, lang))
        compare_audit(audit, reading)
        return audit

    async def _audit_llm(self, frame: ClaimFrame, evidence: EvidenceItemFrame) -> EvidenceAudit:
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
