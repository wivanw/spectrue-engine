"""Evidence audit use cases."""

from __future__ import annotations

from spectrue_core.adapters.llm.evidence_audit import EvidenceAuditSkill
from spectrue_core.domain.evidence.audit import run_evidence_audit


def run_audit(*, claim_frames, llm_client, error_status):
    skill = EvidenceAuditSkill(llm_client)
    return run_evidence_audit(
        claim_frames=claim_frames,
        audit_fn=skill.audit,
        error_status=error_status,
    )