"""Evidence audit use cases."""

from __future__ import annotations

from spectrue_core.domain.evidence.audit import run_evidence_audit


def run_audit(*, claim_frames, audit_fn, error_status):
    return run_evidence_audit(
        claim_frames=claim_frames,
        audit_fn=audit_fn,
        error_status=error_status,
    )
