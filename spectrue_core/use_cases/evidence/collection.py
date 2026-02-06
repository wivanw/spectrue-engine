"""Evidence collection use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.domain.verification.verdict.model import EvidenceFlowInput
from .flow_logic import collect_evidence


def collect_evidence_packs(*, agent: Any, search_mgr: Any, inp: EvidenceFlowInput, claims: list[dict], sources: list[dict]):
    """Collect evidence packs via the legacy evidence flow."""
    return collect_evidence(
        agent=agent,
        search_mgr=search_mgr,
        inp=inp,
        claims=claims,
        sources=sources,
    )
