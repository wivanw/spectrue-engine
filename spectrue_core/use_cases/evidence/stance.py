"""Stance annotation use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.domain.verification.verdict.model import EvidenceFlowInput
from .flow_logic import annotate_evidence_stance


async def annotate_stance(*, agent: Any, inp: EvidenceFlowInput, claims: list[dict], sources: list[dict]) -> list[dict]:
    """Annotate evidence stance using the LLM clustering adapter."""
    return await annotate_evidence_stance(agent=agent, inp=inp, claims=claims, sources=sources)
