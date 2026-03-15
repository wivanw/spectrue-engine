"""Evidence audit orchestration helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


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
    max_concurrency: int | None = None,
) -> EvidenceAuditResult:
    tasks = _collect_tasks(claim_frames)
    if not tasks:
        return EvidenceAuditResult(audits=[], errors={})

    audits: list[Any] = []
    errors: dict[str, dict[str, Any]] = {}
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency is not None and max_concurrency > 0 else None

    async def audit_one(frame: Any, evidence: Any) -> tuple[str, Any, Any]:
        if sem is not None:
            async with sem:
                try:
                    audit = await audit_fn(frame, evidence)
                    return ("ok", evidence, audit)
                except Exception as exc:
                    return ("error", evidence, exc)
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
