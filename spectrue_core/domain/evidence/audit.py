"""Evidence audit orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
import asyncio


AuditCallable = Callable[[Any, Any], Awaitable[Any]]


@dataclass(frozen=True)
class EvidenceAuditResult:
    audits: list[Any]
    errors: dict[str, dict[str, Any]]


def _collect_tasks(claim_frames: list[Any]) -> list[tuple[Any, Any]]:
    tasks: list[tuple[Any, Any]] = []
    for frame in claim_frames:
        for evidence in getattr(frame, "evidence_items", ()) or ():
            tasks.append((frame, evidence))
    return tasks


async def run_evidence_audit(
    claim_frames: list[Any],
    audit_fn: AuditCallable,
    error_status: Any,
) -> EvidenceAuditResult:
    tasks = _collect_tasks(claim_frames)
    if not tasks:
        return EvidenceAuditResult(audits=[], errors={})

    audits: list[Any] = []
    errors: dict[str, dict[str, Any]] = {}

    async def audit_one(frame, evidence):
        try:
            audit = await audit_fn(frame, evidence)
            return ("ok", evidence, audit)
        except Exception as exc:
            return ("error", evidence, exc)

    results = await asyncio.gather(
        *[audit_one(frame, evidence) for frame, evidence in tasks],
        return_exceptions=False,
    )

    for status, evidence, payload in results:
        if status == "ok":
            audits.append(payload)
        else:
            errors[str(getattr(evidence, "evidence_id", ""))] = {
                "status": error_status,
                "error_type": "audit_failed",
                "message": str(payload),
            }

    return EvidenceAuditResult(audits=audits, errors=errors)
