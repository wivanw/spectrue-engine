# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Apply the Decisions metadata sensor to extracted claims (M146 T031/T032).

Per field, the sensor value replaces today's value only when that field's sensor
is calibrated and its expected error is no larger than the measured error of the
current path (`DecisionSensorCalibration.reference_error_rate`) — the same rule
as the inline-source gate. Uncalibrated fields and shadow mode change nothing;
shadow traces the per-field comparison (`metadata_sensor.compare`) that the
calibration in M146 T032 is fitted on.

Values:
- verification_target, claim_role: MAP choice;
- harm: expected level on 1–5 (`score` is the probability-weighted 0-based level);
- importance: expected level mapped to [0, 1] as score / 4;
- check_oracle, time_sensitive, satire: MAP of the predicate; satire → verification_target none.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Mapping, Sequence

from spectrue_core.adapters.llm.decision_gates import SensorReading, gate_mode
from spectrue_core.adapters.llm.metadata_sensor import read_claim_metadata
from spectrue_core.domain.claims import fields
from spectrue_core.domain.claims.model import ClaimRole, VerificationTarget
from spectrue_core.domain.verification.calibration.decision_fusion import (
    SensorCalibration, calibrated_probability,
)
from spectrue_core.runtime_config import DecisionsConfig, DecisionSensorCalibration
from spectrue_core.utils.trace import Trace

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")


def local_window(text: str, claim_text: str) -> str:
    """The sentence holding the claim plus one sentence on each side (whole text if not found)."""
    if not text:
        return claim_text
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    probe = (claim_text or "")[:40]
    for i, s in enumerate(sentences):
        if probe and probe in s:
            return " ".join(sentences[max(0, i - 1): i + 2])
    return text[:1200]


def sensor_values(reading: SensorReading) -> dict[str, tuple[Any, float | None]]:
    """(value, confidence used for acceptance) per answered field; a refused field is absent."""
    answers = reading.result.answers
    out: dict[str, tuple[Any, float | None]] = {}
    for name in ("verification_target", "claim_role"):
        if name in answers:
            out[name] = (answers[name].choice, answers[name].top_confidence)
    if "harm" in answers:
        out["harm"] = ((answers["harm"].score or 0.0) + 1.0, answers["harm"].top_confidence)
    if "importance" in answers:
        out["importance"] = ((answers["importance"].score or 0.0) / 4.0, answers["importance"].top_confidence)
    for name in ("check_oracle", "time_sensitive", "satire"):
        if name in answers:
            out[name] = ((answers[name].probability or 0.0) >= 0.5, answers[name].top_confidence)
    return out


def _accepted(cal: DecisionSensorCalibration, confidence: float | None) -> bool:
    sc = SensorCalibration(weight=cal.weight, bias=cal.bias, reference_error_rate=cal.reference_error_rate)
    if confidence is None or sc.reference_error_rate is None:
        return False
    q = calibrated_probability(confidence, sc)
    return q is not None and (1.0 - q) <= sc.reference_error_rate


_CURRENT = {
    "verification_target": lambda c: fields.enum_value(c, "verification_target"),
    "claim_role": lambda c: fields.enum_value(c, "claim_role"),
    "harm": lambda c: fields.number(c, "harm_potential"),
    "importance": lambda c: fields.unit(c, "importance"),
    "check_oracle": lambda c: bool(c.get("check_oracle")) if isinstance(c, dict) else None,
    "time_sensitive": lambda c: fields.raw(c, "time_sensitive"),
    "satire": lambda c: (fields.unit(c, "satire_likelihood") or 0.0) >= 0.5,
}


def _write(claim: dict, field: str, value: Any) -> None:
    md = claim.get("metadata")
    if field == "verification_target":
        claim["verification_target"] = value
        if md is not None:
            md.verification_target = VerificationTarget(value)
    elif field == "claim_role":
        claim["claim_role"] = claim["role"] = value
        if md is not None:
            md.claim_role = ClaimRole(value)
    elif field == "harm":
        claim["harm_potential"] = value
    elif field == "importance":
        claim["importance"] = value
    elif field == "check_oracle":
        claim["check_oracle"] = bool(value)
    elif field == "time_sensitive":
        if md is not None:
            md.time_sensitive = bool(value)
    elif field == "satire" and value:
        _write(claim, "verification_target", VerificationTarget.NONE.value)


async def apply_metadata_sensor(
    llm_client: Any,
    decisions: DecisionsConfig,
    sensors: Mapping[str, DecisionSensorCalibration],
    claims: Sequence[dict],
    *,
    text: str,
    lang: str,
) -> int:
    """Read the sensor for every claim; apply accepted fields. Returns fields applied."""
    if gate_mode(decisions, "metadata") == "off" or not claims:
        return 0
    if gate_mode(decisions, "retrieval_plan") == "on":
        # The retrieval-plan sensor already asked these questions and its answers are the claim's fields.
        return 0
    readings = await asyncio.gather(*(
        read_claim_metadata(
            llm_client, decisions,
            claim_text=str(c.get("normalized_text") or c.get("text") or ""),
            window=local_window(text, str(c.get("text") or "")),
            lang=lang, claim_id=str(c.get("id")),
        )
        for c in claims
    ))
    applied = 0
    for claim, reading in zip(claims, readings):
        if reading is None:
            continue
        values = sensor_values(reading)
        compare: dict[str, Any] = {}
        for field, (value, conf) in values.items():
            current = _CURRENT[field](claim)
            take = reading.applied and _accepted(sensors[f"meta_{field}"], conf)
            compare[field] = {"sensor": value, "current": current, "confidence": conf, "applied": take}
            if take:
                _write(claim, field, value)
                applied += 1
        Trace.event("metadata_sensor.compare", {"claim_id": claim.get("id"), "locale": lang, "fields": compare})
    return applied
