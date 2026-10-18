# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""M145 R028: sensor reader, Bayesian fusion and the gate decision rules. No network."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.domain.verification.calibration.decision_fusion import (
    SensorCalibration, any_of, calibrated_probability, expected_error, fuse, logit,
)
from spectrue_core.llm.decisions import DecisionAnswer, DecisionQuestion, DecisionResult
from spectrue_core.llm.errors import LLMCallError, LLMFailureKind
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors
from spectrue_core.scoring.budget_allocation import ExtractBudgetParams, select_extracts_by_evoi
from spectrue_core.use_cases.retrieval import decision_gates as uc

Q = [DecisionQuestion.predicate("x", "?")]


def _res(**probs) -> DecisionResult:
    return DecisionResult(answers={k: DecisionAnswer(name=k, type="predicate", probability=v)
                                   for k, v in probs.items()}, model="m", usage={"input_tokens": 10})


def _refusal(name="x") -> DecisionResult:
    return DecisionResult(answers={name: DecisionAnswer(name=name, type="refusal")}, model="m", usage={})


def _client(result=None, exc=None):
    c = MagicMock()
    c.call_decision = AsyncMock(return_value=result, side_effect=exc)
    return c


async def _read(mode, client, **cfg):
    with patch("spectrue_core.adapters.llm.decision_gates.Trace") as tr:
        r = await read_sensor(gate="g", cfg=DecisionsConfig(gate_modes={"g": mode}, **cfg),
                              llm_client=client, input_text="in", questions=Q)
    return r, {c.args[0]: c.args[1] for c in tr.event.call_args_list}


# ---------------- fusion math ----------------

def test_uncalibrated_sensor_changes_nothing():
    cal = SensorCalibration()
    assert fuse(0.3, 0.99, cal) == 0.3
    assert cal.delta(0.99) == 0.0
    assert calibrated_probability(0.99, cal) is None


def test_fuse_is_log_odds_addition():
    cal = SensorCalibration(weight=1.0, bias=0.0)
    post = fuse(0.5, 0.8, cal)
    assert logit(post) == pytest.approx(logit(0.5) + logit(0.8))


def test_any_of_and_expected_error():
    assert any_of([0.5, 0.5]) == pytest.approx(0.75)
    assert any_of([]) == 0.0
    assert expected_error(0.9) == pytest.approx(0.1)


# ---------------- EVOI candidate selection ----------------

def test_select_extracts_by_evoi_uses_budget_rule():
    p = ExtractBudgetParams()
    mask = select_extracts_by_evoi([0.1, 0.95, 0.9, 0.2, 0.05], params=p)
    # min_extracts always taken (highest first), then EVOI = q * decay^k vs max(min_evoi, cost)
    taken = [q for q, m in zip([0.1, 0.95, 0.9, 0.2, 0.05], mask) if m]
    assert taken[:2] == [0.95, 0.9] or sorted(taken, reverse=True)[:2] == [0.95, 0.9]
    floor = max(p.min_evoi_threshold, p.marginal_cost_per_extract)
    for k, q in enumerate(sorted(taken, reverse=True)):
        if k >= p.min_extracts:
            assert p.value_uncertainty * q * p.diminishing_returns_decay ** k >= floor


def test_select_extracts_respects_prior_extracts():
    p = ExtractBudgetParams()
    assert select_extracts_by_evoi([0.9], params=p, extracts_used=p.max_extracts) == [False]


# ---------------- sensor reader ----------------

@pytest.mark.asyncio
async def test_off_makes_no_call():
    c = _client(_res(x=0.9))
    r, ev = await _read("off", c)
    assert r is None and ev == {}
    c.call_decision.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_is_internal_and_not_applied():
    c = _client(_res(x=0.9))
    r, ev = await _read("shadow", c)
    assert r.applied is False and r.probability("x") == 0.9
    assert c.call_decision.call_args.kwargs["billing"] == "internal"
    assert "decision_gate.reading" in ev


@pytest.mark.asyncio
async def test_on_is_user_billed_and_applied():
    c = _client(_res(x=0.9))
    r, _ = await _read("on", c)
    assert r.applied is True
    assert c.call_decision.call_args.kwargs["billing"] == "user"


@pytest.mark.asyncio
async def test_refusal_is_unusable_not_a_label():
    r, ev = await _read("on", _client(_refusal()))
    assert r is None and ev["decision_gate.non_ok"]["outcome"] == "refusal"


@pytest.mark.asyncio
async def test_independent_refusal_drops_only_that_answer():
    res = DecisionResult(answers={"x": DecisionAnswer(name="x", type="predicate", probability=0.9),
                                  "y": DecisionAnswer(name="y", type="refusal")}, model="m", usage={})
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        r = await read_sensor(gate="g", cfg=DecisionsConfig(gate_modes={"g": "on"}), llm_client=_client(res),
                              input_text="in", questions=Q, independent=True)
    assert set(r.result.answers) == {"x"}


@pytest.mark.asyncio
async def test_independent_all_refused_is_unusable():
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        r = await read_sensor(gate="g", cfg=DecisionsConfig(gate_modes={"g": "on"}), llm_client=_client(_refusal()),
                              input_text="in", questions=Q, independent=True)
    assert r is None


@pytest.mark.asyncio
async def test_provider_failure_is_typed():
    r, ev = await _read("on", _client(exc=LLMCallError(message="x", kind=LLMFailureKind.RATE_LIMITED)))
    assert r is None and ev["decision_gate.non_ok"]["outcome"] == "rate_limited"


@pytest.mark.asyncio
async def test_internal_bug_propagates():
    with pytest.raises(RuntimeError):
        await _read("on", _client(exc=RuntimeError("bug")))


@pytest.mark.asyncio
async def test_input_over_cap_never_calls():
    c = _client(_res(x=0.9))
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        r = await read_sensor(gate="g", cfg=DecisionsConfig(gate_modes={"g": "on"}, max_input_chars=500),
                              llm_client=c, input_text="x" * 501, questions=Q)
    assert r is None
    c.call_decision.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_mode_is_wiring_error():
    with pytest.raises(ValueError):
        await _read(MagicMock(), _client(_res(x=0.9)))


# ---------------- gate rules ----------------

def _reading(applied=True, **probs) -> SensorReading:
    return SensorReading(result=_res(**probs), applied=applied, latency_ms=1)


def _sensors(**over) -> dict:
    s = uncalibrated_sensors()
    for k, (w, b, ref) in over.items():
        s[k] = DecisionSensorCalibration(version="t", weight=w, bias=b, reference_error_rate=ref)
    return s


def _agent(**readers):
    return SimpleNamespace(relevance_skill=SimpleNamespace(**{k: AsyncMock(return_value=v) for k, v in readers.items()}))


CLAIMS = [{"id": "c1", "text": "a", "check_worthiness": 0.8}, {"id": "c2", "text": "b", "check_worthiness": 0.6}]


@pytest.mark.asyncio
async def test_g2_scales_worthiness_by_calibrated_probability():
    agent = _agent(read_verifiability=_reading(checkable=0.9))
    s = _sensors(verifiability=(1.0, 0.0, None))
    out = await uc.apply_verifiability(agent, DecisionsConfig(gate_modes={"verifiability": "on"}), s, CLAIMS, "en")
    assert out[0]["check_worthiness"] == pytest.approx(0.8 * 0.9)
    assert len(out) == 2  # nothing dropped: target-selection EVOI decides


@pytest.mark.asyncio
async def test_g2_uncalibrated_or_shadow_changes_nothing():
    on = DecisionsConfig(gate_modes={"verifiability": "on"})
    assert await uc.apply_verifiability(_agent(read_verifiability=_reading(checkable=0.01)),
                                        on, uncalibrated_sensors(), CLAIMS, "en") is None
    shadow = DecisionsConfig(gate_modes={"verifiability": "shadow"})
    assert await uc.apply_verifiability(_agent(read_verifiability=_reading(False, checkable=0.01)),
                                        shadow, _sensors(verifiability=(1.0, 0.0, None)), CLAIMS, "en") is None


@pytest.mark.asyncio
async def test_g4_adds_calibrated_log_odds_only_when_applied():
    s = _sensors(search_stop=(1.0, 0.0, None))
    on = DecisionsConfig(gate_modes={"search_stop": "on"})
    d = await uc.sufficiency_sensor_log_odds(_agent(read_sufficiency=_reading(sufficient=0.8)), on, s,
                                             CLAIMS[0], [], lang="en", claim_id="c1")
    assert d == pytest.approx(math.log(0.8 / 0.2))
    shadow = DecisionsConfig(gate_modes={"search_stop": "shadow"})
    assert await uc.sufficiency_sensor_log_odds(_agent(read_sufficiency=_reading(False, sufficient=0.8)), shadow, s,
                                                CLAIMS[0], [], lang="en", claim_id="c1") == 0.0


def test_g4_sensor_term_cannot_make_no_sources_sufficient():
    from spectrue_core.use_cases.claims.sufficiency import SufficiencyStatus, check_sufficiency_for_claim
    assert check_sufficiency_for_claim(CLAIMS[0], [], sensor_log_odds=50.0).status == SufficiencyStatus.INSUFFICIENT


@pytest.mark.asyncio
async def test_g3_filter_selects_by_evoi_and_stores_evidence():
    url_metadata = {f"u{i}": {"title": "t", "snippet": "s", "score": 0.5} for i in range(6)}
    probs = iter([0.99, 0.98, 0.97, 0.01, 0.01, 0.01])
    skill = SimpleNamespace(read_evidence=AsyncMock(side_effect=lambda *a, **k: _reading(c0=next(probs))))
    f = uc.build_extract_filter(
        SimpleNamespace(relevance_skill=skill), DecisionsConfig(gate_modes={"extract": "on"}),
        _sensors(extract=(1.0, 0.0, None)), url_metadata=url_metadata,
        seen_by_claims=lambda u: {"c1"}, claim_id_map={"c1": CLAIMS[0]}, lang="en",
    )
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        kept = await f(list(url_metadata))
    assert set(kept) <= {"u0", "u1", "u2"} and len(kept) >= ExtractBudgetParams().min_extracts
    assert url_metadata["u0"]["decision_evidence"] == {"c1": 0.99}


@pytest.mark.asyncio
async def test_g3_shadow_extracts_everything():
    url_metadata = {"u0": {"score": 0.5}, "u1": {"score": 0.5}, "u2": {"score": 0.5}}
    skill = SimpleNamespace(read_evidence=AsyncMock(return_value=_reading(False, c0=0.01)))
    f = uc.build_extract_filter(
        SimpleNamespace(relevance_skill=skill), DecisionsConfig(gate_modes={"extract": "shadow"}),
        _sensors(extract=(1.0, 0.0, None)), url_metadata=url_metadata,
        seen_by_claims=lambda u: {"c1"}, claim_id_map={"c1": CLAIMS[0]}, lang="en",
    )
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        assert await f(["u0", "u1", "u2"]) == ["u0", "u1", "u2"]


@pytest.mark.asyncio
async def test_g3_on_uncalibrated_extracts_everything():
    """Six-fact run 2026-10-18: uncalibrated, the EVOI cut on similarity alone dropped the page with the answer."""
    url_metadata = {f"u{i}": {"score": 0.5} for i in range(9)}
    skill = SimpleNamespace(read_evidence=AsyncMock(return_value=_reading(True, c0=0.9)))
    f = uc.build_extract_filter(
        SimpleNamespace(relevance_skill=skill), DecisionsConfig(gate_modes={"extract": "on"}),
        uncalibrated_sensors(), url_metadata=url_metadata,
        seen_by_claims=lambda u: {"c1"}, claim_id_map={"c1": CLAIMS[0]}, lang="en",
    )
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        assert await f(list(url_metadata)) == list(url_metadata)


def test_g3_off_has_no_filter():
    assert uc.build_extract_filter(None, DecisionsConfig(), uncalibrated_sensors(), url_metadata={},
                                   seen_by_claims=set, claim_id_map={}, lang="en") is None


@pytest.mark.asyncio
async def test_g1_fuses_similarity_and_reuses_g3_evidence():
    skill = SimpleNamespace(read_evidence=AsyncMock())
    entries = [("c1", [{"url": "u1", "similarity_score": 0.5, "decision_evidence": {"c1": 0.9}}])]
    out = await uc.fuse_result_similarity(
        SimpleNamespace(relevance_skill=skill), DecisionsConfig(gate_modes={"search_relevance": "on"}),
        _sensors(search_relevance=(1.0, 0.0, None)), entries, {"c1": CLAIMS[0]}, "en")
    assert out[("c1", "u1")] == pytest.approx(0.9)
    skill.read_evidence.assert_not_called()


def test_g5_uses_sensor_only_within_reference_error():
    s = _sensors(inline_relevant=(1.0, 0.0, 0.05), inline_primary=(1.0, 0.0, 0.05))
    assert uc.inline_labels_from_reading(_reading(is_relevant=0.99, is_primary=0.02), s) == (True, False)
    assert uc.inline_labels_from_reading(_reading(is_relevant=0.99, is_primary=0.2), s) is None  # 0.2 > 0.05
    assert uc.inline_labels_from_reading(_reading(is_relevant=0.99, is_primary=0.02), uncalibrated_sensors()) is None
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        assert uc.inline_labels_from_reading(_reading(False, is_relevant=0.99, is_primary=0.02), s) is None
