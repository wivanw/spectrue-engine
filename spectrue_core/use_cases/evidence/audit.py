"""Evidence audit use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.evidence_audit import EvidenceAuditSkill
from spectrue_core.domain.evidence.audit import run_evidence_audit


def run_audit(
    *,
    claim_frames: list,
    llm_client: Any,
    error_status: Any,
    max_concurrency: int | None = None,
) -> Any:
    skill = EvidenceAuditSkill(llm_client)
    return run_evidence_audit(
        claim_frames=claim_frames,
        audit_fn=skill.audit,
        error_status=error_status,
        max_concurrency=max_concurrency,
    )