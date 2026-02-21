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

from typing import Any
from spectrue_core.llm.llm_client import LLMClient
from spectrue_core.agents.llm_schemas import CLAIM_AUDIT_SCHEMA
from spectrue_core.adapters.llm.audit_prompts import (
    build_claim_audit_prompt,
    build_claim_audit_system_prompt,
)
from spectrue_core.domain.claims.frame import ClaimFrame
from spectrue_core.domain.verification.verdict.model import ClaimAudit
from spectrue_core.utils.trace import Trace
from spectrue_core.llm.model_registry import ModelID


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
                model=ModelID.NANO,
                temperature=0,
                fail_on_schema_error=False,
            )
            
            # Salvage missing fields to avoid pydantic/dataclass validation errors
            salvaged = self._salvage_audit_response(response, frame.claim_id)
            audit = ClaimAudit(**salvaged)
            
            if audit.claim_id != frame.claim_id:
                raise ValueError("claim_id mismatch in audit response")
            Trace.event("claim_audit.complete", {"claim_id": frame.claim_id})
            return audit
        except Exception as exc:
            Trace.event("claim_audit.error", {"claim_id": frame.claim_id, "error": str(exc)})
            raise

    def _salvage_audit_response(self, response: dict[str, Any], claim_id: str) -> dict[str, Any]:
        """Fill in missing fields with safe defaults to avoid validation crashes."""
        out = dict(response or {})
        
        # 1. Identity
        if not out.get("claim_id"):
            out["claim_id"] = claim_id
            
        # 2. Categorization
        if out.get("predicate_type") not in ("event", "measurement", "quote", "policy", "ranking", "causal", "other"):
            out["predicate_type"] = "other"
            
        # 3. Strength & Confidence
        if out.get("assertion_strength") not in ("weak", "medium", "strong"):
            out["assertion_strength"] = "medium"
            
        if not isinstance(out.get("audit_confidence"), (int, float)):
            out["audit_confidence"] = 0.5
            
        # 4. List fields
        list_fields = [
            "truth_conditions", "expected_evidence_types", "failure_modes",
            "risk_facets", "honesty_facets", "what_would_change_mind"
        ]
        for field_name in list_fields:
            if not isinstance(out.get(field_name), list):
                out[field_name] = []
                
        return out
