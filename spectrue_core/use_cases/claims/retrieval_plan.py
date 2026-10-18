# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build the call-B-shaped retrieval plan from call A + the Decisions sensor (M147 T022).

Every field is either a sensor reading or derived deterministically from one, so the claim
builder downstream is unchanged:
- check_worthiness = P(verification_target != none)  (M146 plan: checkability)
- claim_category   = SATIRE if P(satire) >= 0.5, OPINION if the target is none, else FACTUAL
- check_oracle     = target != none: asking the fact-checker database costs nothing, so the
                     EVOI of the query is positive for every checkable claim
- metadata_confidence = the lowest answer confidence, quantised to the schema's three levels
A refused question leaves its field out; the sanitizer's defaults then apply as for call B.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading


def _confidence_level(p: float) -> str:
    return "high" if p >= 0.8 else "medium" if p >= 0.5 else "low"


def plan_from_reading(reading: SensorReading, *, normalized_text: str, search_queries: list[str], lang: str) -> dict:
    a = reading.result.answers
    out: dict[str, Any] = {
        "search_queries": list(search_queries or []),
        "query_candidates": [],
        "search_locale_plan": {"primary": (lang or "en")[:2], "fallback": ["en"]},
    }
    vt = a.get("verification_target")
    if vt is not None and vt.choice:
        out["verification_target"] = vt.choice
        p_none = float((vt.probabilities or {}).get("none", 1.0 if vt.choice == "none" else 0.0))
        out["check_worthiness"] = round(1.0 - p_none, 4)
        out["check_oracle"] = vt.choice != "none"
    if "claim_role" in a and a["claim_role"].choice:
        out["claim_role"] = a["claim_role"].choice
    if "harm" in a:
        out["harm_potential"] = int(round((a["harm"].score or 0.0) + 1))
    if "importance" in a:
        out["importance"] = round((a["importance"].score or 0.0) / 4.0, 4)
    p_satire = float(a["satire"].probability or 0.0) if "satire" in a else 0.0
    out["satire_likelihood"] = round(p_satire, 4)
    out["claim_category"] = ("SATIRE" if p_satire >= 0.5
                             else "OPINION" if out.get("verification_target") == "none" else "FACTUAL")
    if "time_sensitive" in a:
        out["time_sensitive"] = bool((a["time_sensitive"].probability or 0.0) >= 0.5)
    for name in ("evidence_need", "search_method"):
        if name in a and a[name].choice:
            out[name] = a[name].choice
    if "needs_primary" in a:
        out["evidence_req"] = {"needs_primary": bool((a["needs_primary"].probability or 0.0) >= 0.5),
                               "needs_2_independent": False}
    if "structure_type" in a and a["structure_type"].choice:
        out["structure"] = {"type": a["structure_type"].choice, "premises": [], "conclusion": normalized_text,
                            "dependencies": []}
    confs = [x.top_confidence for x in a.values() if x.top_confidence is not None]
    out["metadata_confidence"] = _confidence_level(min(confs)) if confs else "low"
    return out


def compare_plans(call_b: dict, sensor_plan: dict) -> dict:
    """Field-by-field agreement for shadow tracing."""
    keys = ("verification_target", "claim_role", "harm_potential", "claim_category", "evidence_need",
            "search_method", "check_oracle", "metadata_confidence")
    return {k: {"call_b": call_b.get(k), "sensor": sensor_plan.get(k)} for k in keys}
