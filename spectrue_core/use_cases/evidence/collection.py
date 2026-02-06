"""Evidence collection use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.verification.evidence.evidence import build_evidence_pack
from spectrue_core.verification.pipeline.pipeline_evidence import EvidenceFlowInput, collect_evidence


def collect_evidence_packs(*, agent: Any, search_mgr: Any, inp: EvidenceFlowInput, claims: list[dict], sources: list[dict]):
    """Collect evidence packs via the legacy evidence flow."""
    return collect_evidence(
        agent=agent,
        search_mgr=search_mgr,
        build_evidence_pack=build_evidence_pack,
        calibration_registry=None,
        inp=inp,
        claims=claims,
        sources=sources,
    )
