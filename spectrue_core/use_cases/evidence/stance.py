"""Stance annotation use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.verification.pipeline.pipeline_evidence import EvidenceFlowInput, annotate_evidence_stance


def annotate_stance(*, agent: Any, inp: EvidenceFlowInput, claims: list[dict], sources: list[dict]) -> list[dict]:
    """Annotate evidence stance using the LLM clustering adapter."""
    return annotate_evidence_stance(agent=agent, inp=inp, claims=claims, sources=sources)
