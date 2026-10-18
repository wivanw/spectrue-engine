# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Decisions gates G1–G5: sensor readings -> decisions (M145 Phase R, R028).

Decisions is a sensor (Constitution II). Every rule here fuses its calibrated
reading into a signal the pipeline already has and lets the step's existing
decision rule act — no thresholds of our own:

- G1 search_relevance: Bayesian update of each result's similarity before rerank.
- G2 verifiability:    check_worthiness *= P(checkable) -> target-selection EVOI.
- G3 extract:          P(useful) of a URL -> `select_extracts_by_evoi`.
- G4 search_stop:      sensor log-odds term in the Bayesian sufficiency posterior.
- G5 inline_source:    replace the LLM label only when the sensor's expected error
                       is no larger than the measured error of the LLM path.

Uncalibrated sensors (weight = bias = 0) change nothing (decision_fusion). Shadow
readings are only traced (`decision_gate.shadow_effect`). Steps stay thin (V).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping, Sequence

from spectrue_core.adapters.llm.decision_gates import SensorReading, gate_mode
from spectrue_core.domain.evidence.source_similarity import similarity_score
from spectrue_core.domain.claims import fields as claim_fields
from spectrue_core.domain.verification.calibration.decision_fusion import (
    SensorCalibration,
    any_of,
    calibrated_probability,
    expected_error,
    fuse,
)
from spectrue_core.runtime_config import DecisionsConfig, DecisionSensorCalibration
from spectrue_core.scoring.budget_allocation import ExtractBudgetParams, select_extracts_by_evoi
from spectrue_core.utils.trace import Trace

Sensors = Mapping[str, DecisionSensorCalibration]


def sensor(sensors: Sensors, name: str) -> SensorCalibration:
    c = sensors[name]  # a missing sensor is a wiring bug: KeyError
    return SensorCalibration(weight=c.weight, bias=c.bias, reference_error_rate=c.reference_error_rate)


def _claim_id(c: Any) -> str | None:
    cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
    return str(cid) if cid is not None else None


def claims_by_id(claims: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in claims:
        cid = _claim_id(c)
        if cid is not None:
            out.setdefault(cid, c)
    return out


def _evidence_probs(reading: SensorReading, n: int) -> list[float]:
    return [reading.probability(f"c{i}") for i in range(n)]


# ---------------- G2: verifiability -> target selection EVOI ----------------

async def apply_verifiability(
    agent: Any, decisions: DecisionsConfig, sensors: Sensors, claims: Sequence[Any], lang: str
) -> list[Any] | None:
    """Claims with `check_worthiness` scaled by P(checkable), or None when unchanged.

    EV(check) = P(checkable) * value, and target selection's EVOI is linear in
    worthiness, so the scaled worthiness is exactly the expected value. No claim
    is dropped here; the Bayesian budget decides which claims get searched.
    """
    if gate_mode(decisions, "verifiability") == "off" or not claims:
        return None
    readings = await asyncio.gather(*(
        agent.relevance_skill.read_verifiability(c, lang=lang, claim_id=_claim_id(c)) for c in claims
    ))
    cal = sensor(sensors, "verifiability")
    out, changed = [], False
    for c, r in zip(claims, readings):
        q = calibrated_probability(r.probability("checkable"), cal) if r else None
        if q is None or not isinstance(c, dict):
            out.append(c)
            continue
        prior = claim_fields.or_default(
            claim_fields.unit(c, "check_worthiness") if claim_fields.unit(c, "check_worthiness") is not None
            else claim_fields.unit(c, "importance"), 0.5)
        posterior = prior * q
        Trace.event("decision_gate.effect" if r.applied else "decision_gate.shadow_effect", {
            "gate": "verifiability", "claim_id": _claim_id(c), "p_checkable": q,
            "check_worthiness_prior": prior, "check_worthiness_posterior": posterior,
        })
        if r.applied:
            out.append({**c, "check_worthiness": posterior})
            changed = True
        else:
            out.append(c)
    return out if changed else None


# ---------------- G1: result similarity (before rerank) ----------------

async def fuse_result_similarity(
    agent: Any,
    decisions: DecisionsConfig,
    sensors: Sensors,
    entries: Sequence[tuple[str | None, list[dict[str, Any]]]],
    claims: Mapping[str, Any],
    lang: str,
) -> dict[tuple[str, str], float] | None:
    """Posterior similarity per (claim_id, url), or None when unchanged.

    The sensor reading is reused from G3 (`decision_evidence`, stored before
    extract) when present; otherwise one reading per result.
    """
    if gate_mode(decisions, "search_relevance") == "off":
        return None
    cal = sensor(sensors, "search_relevance")
    applied = decisions.mode("search_relevance") == "on"
    jobs: list[tuple[str, dict[str, Any]]] = [
        (cid, src) for cid, payload in entries if cid in claims for src in payload
        if isinstance(src, dict) and (src.get("url") or src.get("link"))
    ]

    async def _p(cid: str, src: dict[str, Any]) -> float | None:
        cached = (src.get("decision_evidence") or {}).get(cid)
        if cached is not None:
            return float(cached)
        r = await agent.relevance_skill.read_evidence(
            [claims[cid]], title=str(src.get("title") or ""),
            snippet=str(src.get("snippet") or src.get("content") or ""),
            gate="search_relevance", lang=lang, trace_meta={"claim_id": cid},
        )
        return r.probability("c0") if r else None

    probs = await asyncio.gather(*(_p(cid, src) for cid, src in jobs))
    out: dict[tuple[str, str], float] = {}
    for (cid, src), p in zip(jobs, probs):
        if p is None or not cal.calibrated:
            continue
        url = str(src.get("url") or src.get("link"))
        prior = similarity_score(src)
        out[(cid, url)] = fuse(prior, p, cal)
    Trace.event("decision_gate.effect" if applied else "decision_gate.shadow_effect", {
        "gate": "search_relevance", "results": len(jobs), "updated": len(out), "calibrated": cal.calibrated,
    })
    return out if applied and out else None


# ---------------- G3: extract selection by EVOI ----------------

def build_extract_filter(
    agent: Any,
    decisions: DecisionsConfig,
    sensors: Sensors,
    *,
    url_metadata: dict[str, dict[str, Any]],
    seen_by_claims: Callable[[str], set[str]],
    claim_id_map: Mapping[str, Any],
    lang: str,
    params: ExtractBudgetParams | None = None,
) -> Callable[[list[str]], Awaitable[list[str]]] | None:
    """An `extract_filter` for `FixedPipelineContext`, or None when the gate is off.

    P(useful | url) = fuse(similarity prior, P(any claim has evidence here)), then
    `select_extracts_by_evoi` decides, counting extracts across stages.
    """
    if gate_mode(decisions, "extract") == "off":
        return None
    cal = sensor(sensors, "extract")
    extracts_used = 0

    async def _quality(url: str) -> tuple[float, bool]:
        meta = url_metadata.get(url) or {}
        cids = [c for c in sorted(seen_by_claims(url)) if c in claim_id_map]
        prior = similarity_score(dict(meta.get("provider_meta") or {}, score=meta.get("score")))
        r = await agent.relevance_skill.read_evidence(
            [claim_id_map[c] for c in cids], title=str(meta.get("title") or ""),
            snippet=str(meta.get("snippet") or ""), gate="extract", lang=lang, trace_meta={"url": url},
        )
        if r is None:
            return prior, False
        probs = _evidence_probs(r, min(len(cids), len(r.result.answers)))
        meta["decision_evidence"] = dict(zip(cids, probs))  # reused by G1
        return fuse(prior, any_of(probs), cal), r.applied

    async def _filter(urls: list[str]) -> list[str]:
        nonlocal extracts_used
        scored = await asyncio.gather(*(_quality(u) for u in urls))
        mask = select_extracts_by_evoi([q for q, _ in scored], params=params, extracts_used=extracts_used)
        # Uncalibrated, the sensor adds nothing to the similarity prior, so the EVOI cut would be a
        # cut on search similarity alone, not a Decisions decision: it is not applied (it dropped the
        # one page with the answer on the 2026-10-18 six-fact run).
        applied = cal.calibrated and any(a for _, a in scored)
        Trace.event("decision_gate.effect" if applied else "decision_gate.shadow_effect", {
            "gate": "extract", "candidates": len(urls), "selected": sum(mask),
            "extracts_used_before": extracts_used, "calibrated": cal.calibrated,
            "qualities": [round(q, 4) for q, _ in scored],
        })
        if not applied:
            extracts_used += len(urls)
            return urls
        keep = [u for u, m in zip(urls, mask) if m]
        extracts_used += len(keep)
        return keep

    return _filter


# ---------------- G4: sensor term in the sufficiency posterior ----------------

async def sufficiency_sensor_log_odds(
    agent: Any, decisions: DecisionsConfig, sensors: Sensors, claim: Any, sources: list[Any],
    *, lang: str, claim_id: str,
) -> float:
    """Log-likelihood term for `check_sufficiency_for_claim` (0.0 = no applied reading)."""
    if gate_mode(decisions, "search_stop") == "off":
        return 0.0
    r = await agent.relevance_skill.read_sufficiency(claim, sources, lang=lang, claim_id=claim_id)
    if r is None:
        return 0.0
    delta = sensor(sensors, "search_stop").delta(r.probability("sufficient"))
    Trace.event("decision_gate.effect" if r.applied else "decision_gate.shadow_effect", {
        "gate": "search_stop", "claim_id": claim_id, "sensor_log_odds": delta,
    })
    return delta if r.applied else 0.0


# ---------------- G5: inline source labels ----------------

def inline_labels_from_reading(reading: SensorReading | None, sensors: Sensors) -> tuple[bool, bool] | None:
    """(is_relevant, is_primary) from the sensor, or None to keep the LLM path.

    Each label is the MAP label of its calibrated probability and is used only when
    its expected error is <= the measured error rate of the LLM path.
    """
    if reading is None:
        return None
    labels: list[bool] = []
    for name, sensor_name in (("is_relevant", "inline_relevant"), ("is_primary", "inline_primary")):
        cal = sensor(sensors, sensor_name)
        q = calibrated_probability(reading.probability(name), cal)
        if q is None or cal.reference_error_rate is None or expected_error(q) > cal.reference_error_rate:
            return None
        labels.append(q >= 0.5)
    if not reading.applied:
        Trace.event("decision_gate.shadow_effect", {"gate": "inline_source", "labels": labels})
        return None
    return labels[0], labels[1]
