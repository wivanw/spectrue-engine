# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply the Decisions stance sensor to clustered evidence rows (M147 T020).

off    -> nothing.
shadow -> one reading per row with a claim; trace `stance_sensor.compare` (matrix vs sensor), change nothing.
on     -> the sensor IS the stance classifier: a row's stance is the sensor's choice (the matrix still maps
          sources to claims and extracts quotes). Quality is gated by the fact-checker benchmark, not per row.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from spectrue_core.adapters.llm.decision_gates import gate_mode
from spectrue_core.adapters.llm.stance_sensor import read_stance
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig
from spectrue_core.utils.trace import Trace

MAX_CONCURRENT_STANCE_READS = 16


def row_text(row: Mapping[str, Any]) -> str:
    """The text the matrix judged: its quote if any, else the source content / snippet."""
    for key in ("quote", "quote_span", "content", "content_excerpt", "snippet"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


async def apply_stance_sensor(
    llm_client: Any,
    decisions: DecisionsConfig,
    sensors: Mapping[str, DecisionSensorCalibration],
    rows: Sequence[dict],
    claims_by_id: Mapping[str, str],
    *,
    lang: str,
) -> int:
    """Returns the number of rows whose stance was changed (0 unless the gate is on and calibrated)."""
    if gate_mode(decisions, "stance") == "off" or not rows:
        return 0
    sem = asyncio.Semaphore(MAX_CONCURRENT_STANCE_READS)
    targets = [r for r in rows if isinstance(r, dict) and claims_by_id.get(str(r.get("claim_id") or "")) and row_text(r)]

    async def one(row: dict):
        async with sem:
            return await read_stance(llm_client, decisions, claim_text=claims_by_id[str(row["claim_id"])],
                                     source_text=row_text(row), lang=lang, claim_id=str(row["claim_id"]),
                                     url=str(row.get("url") or ""))

    readings = await asyncio.gather(*(one(r) for r in targets))
    changed = 0
    agree = total = 0
    for row, reading in zip(targets, readings):
        if reading is None:
            continue
        answer = reading.result.answers.get("stance")
        if answer is None or answer.choice is None:
            continue
        matrix = str(row.get("stance") or "").lower()
        total += 1
        agree += answer.choice == matrix
        take = reading.applied and answer.choice != matrix
        Trace.event("stance_sensor.compare", {
            "claim_id": row.get("claim_id"), "url": str(row.get("url") or "")[:160], "matrix": matrix,
            "sensor": answer.choice, "probabilities": answer.probabilities, "confidence": answer.top_confidence,
            "applied": take,
        })
        if take:
            row["stance_matrix"] = matrix
            row["stance"] = answer.choice
            changed += 1
    Trace.event("stance_sensor.summary", {"rows": len(rows), "read": total, "agree": agree, "changed": changed})
    return changed
