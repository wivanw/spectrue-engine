# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Oracle validation result from the Decisions sensor (M147 T025). Same shape as validate_batch."""

from __future__ import annotations

from spectrue_core.adapters.llm.decision_gates import SensorReading


def oracle_result_from_reading(reading: SensorReading, n_candidates: int, *, jackpot_threshold: float) -> dict:
    a = reading.result.answers
    best = a.get("best")
    if best is None or best.choice in (None, "none") or not str(best.choice).isdigit() or int(best.choice) >= n_candidates:
        return {"best_index": -1, "relevance_score": 0.0, "status": "EMPTY", "reasoning": "sensor: no match",
                "is_jackpot": False}
    match = a.get("match")
    relevance = round((match.score or 0.0) / 4.0, 4) if match is not None else 0.0  # levels 1..5 -> 0..1
    status = (a["status"].choice or "mixed").upper() if "status" in a else "MIXED"
    rating = a.get("rating")
    harm = a.get("harm")
    return {
        "best_index": int(best.choice),
        "relevance_score": relevance,
        "status": status,
        "verified_score": round((rating.score or 0.0) / 4.0, 4) if rating is not None else -1.0,
        "danger_score": round((harm.score or 0.0) / 4.0, 4) if harm is not None else -1.0,
        "reasoning": "decisions sensor",
        "is_jackpot": relevance > jackpot_threshold,
    }


def compare_oracle(llm: dict, sensor: dict) -> dict:
    keys = ("best_index", "relevance_score", "status", "verified_score", "is_jackpot")
    return {k: {"llm": llm.get(k), "sensor": sensor.get(k)} for k in keys}
