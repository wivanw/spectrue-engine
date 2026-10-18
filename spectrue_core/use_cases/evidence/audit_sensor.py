# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build / compare the deep-mode evidence audit from the Decisions sensor (M147 T021)."""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading
from spectrue_core.adapters.llm.evidence_audit_sensor import FIELDS
from spectrue_core.schema.rgba_audit import EvidenceAudit
from spectrue_core.utils.trace import Trace


def audit_from_reading(frame: Any, evidence: Any, reading: SensorReading) -> EvidenceAudit:
    """A field the sensor refused (independent questions) is the schema's 'cannot tell' value."""
    answers = reading.result.answers
    unknown = {"stance": "unclear", "directness": "tangential", "specificity": "low",
               "quote_integrity": "not_applicable", "novelty_vs_copy": "unknown"}
    values = {f: (answers[f].choice if f in answers and answers[f].choice in FIELDS[f] else unknown[f]) for f in FIELDS}
    stance_conf = answers["stance"].top_confidence if "stance" in answers else None
    confs = [a.top_confidence for a in answers.values() if a.top_confidence is not None]
    return EvidenceAudit(
        claim_id=frame.claim_id, evidence_id=evidence.evidence_id, source_id=evidence.source_id or evidence.url,
        dependency_hints=[],
        extraction_confidence=float(stance_conf) if stance_conf is not None else 0.0,
        audit_confidence=float(min(confs)) if confs else 0.0,
        **values,
    )


def compare_audit(audit: Any, reading: SensorReading | None) -> None:
    if reading is None:
        return
    answers = reading.result.answers
    Trace.event("evidence_audit_sensor.compare", {
        "claim_id": audit.claim_id, "evidence_id": audit.evidence_id,
        "fields": {f: {"llm": getattr(audit, f, None), "sensor": answers[f].choice if f in answers else None,
                       "confidence": answers[f].top_confidence if f in answers else None} for f in FIELDS},
    })
