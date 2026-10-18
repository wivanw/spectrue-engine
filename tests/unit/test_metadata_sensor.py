# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""M146 T031: Decisions metadata sensor — questions, acceptance rule, writes. No network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.metadata_sensor import CLAIM_ROLES, VERIFICATION_TARGETS, metadata_questions
from spectrue_core.domain.claims.model import ClaimMetadata, VerificationTarget
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors
from spectrue_core.use_cases.claims.metadata_sensor import apply_metadata_sensor, local_window


def _result(vt="none", vt_conf=0.97, harm_score=3.0, imp_score=2.0, satire=0.02) -> DecisionResult:
    a = {
        "verification_target": DecisionAnswer(name="verification_target", type="choice", choice=vt, confidence=vt_conf),
        "claim_role": DecisionAnswer(name="claim_role", type="choice", choice="core", confidence=0.9),
        "harm": DecisionAnswer(name="harm", type="score", score=harm_score, confidence=0.8),
        "importance": DecisionAnswer(name="importance", type="score", score=imp_score, confidence=0.8),
        "check_oracle": DecisionAnswer(name="check_oracle", type="predicate", probability=0.1),
        "time_sensitive": DecisionAnswer(name="time_sensitive", type="predicate", probability=0.9),
        "satire": DecisionAnswer(name="satire", type="predicate", probability=satire),
    }
    return DecisionResult(answers=a, model="m", usage={"input_tokens": 120})


def _claim():
    return {"id": "c1", "text": "Berlin is the capital of Germany.", "verification_target": "reality",
            "claim_role": "support", "importance": 0.4, "harm_potential": 1,
            "metadata": ClaimMetadata(verification_target=VerificationTarget.REALITY)}


def _calibrated(**fields_ref):
    s = uncalibrated_sensors()
    for f, ref in fields_ref.items():
        s[f"meta_{f}"] = DecisionSensorCalibration(version="t", weight=1.0, reference_error_rate=ref)
    return s


def test_questions_cover_every_field_in_every_locale():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        qs = {q.name: q.payload for q in metadata_questions(lang)}
        assert set(qs) == {"verification_target", "claim_role", "harm", "importance",
                           "check_oracle", "time_sensitive", "satire"}
        assert [c["value"] for c in qs["verification_target"]["choices"]] == list(VERIFICATION_TARGETS)
        assert [c["value"] for c in qs["claim_role"]["choices"]] == list(CLAIM_ROLES)
        assert len(qs["harm"]["levels"]) == 5 and len(qs["importance"]["levels"]) == 5
        assert not any("not found" in str(v) for v in qs.values())


def test_local_window_is_sentence_plus_neighbours():
    text = "One fact. Berlin is the capital of Germany. Third sentence. Fourth."
    assert local_window(text, "Berlin is the capital of Germany.") == "One fact. Berlin is the capital of Germany. Third sentence."


async def _run(mode, sensors, result):
    client = MagicMock()
    client.call_decision = AsyncMock(return_value=result)
    claims = [_claim()]
    with patch("spectrue_core.use_cases.claims.metadata_sensor.Trace") as tr, \
         patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        n = await apply_metadata_sensor(client, DecisionsConfig(gate_modes={"metadata": mode}), sensors,
                                        claims, text="Berlin is the capital of Germany.", lang="en")
    return n, claims[0], tr


@pytest.mark.asyncio
async def test_off_makes_no_call():
    client = MagicMock(); client.call_decision = AsyncMock()
    assert await apply_metadata_sensor(client, DecisionsConfig(), uncalibrated_sensors(), [_claim()],
                                       text="x", lang="en") == 0
    client.call_decision.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_traces_but_changes_nothing():
    n, claim, tr = await _run("shadow", _calibrated(verification_target=0.05), _result())
    assert n == 0 and claim["verification_target"] == "reality"
    cmp = tr.event.call_args.args[1]["fields"]["verification_target"]
    assert cmp == {"sensor": "none", "current": "reality", "confidence": 0.97, "applied": False}


@pytest.mark.asyncio
async def test_on_uncalibrated_changes_nothing():
    n, claim, _ = await _run("on", uncalibrated_sensors(), _result())
    assert n == 0 and claim["importance"] == 0.4


@pytest.mark.asyncio
async def test_on_calibrated_fields_within_reference_error_are_applied():
    n, claim, _ = await _run("on", _calibrated(verification_target=0.05, harm=0.3, importance=0.3), _result())
    assert claim["verification_target"] == "none"
    assert claim["metadata"].verification_target == VerificationTarget.NONE  # both copies
    assert claim["harm_potential"] == 4.0          # score 3 (0-based) -> level 4
    assert claim["importance"] == 0.5              # score 2 / 4
    assert claim["claim_role"] == "support"        # claim_role uncalibrated -> unchanged
    assert n == 3


@pytest.mark.asyncio
async def test_on_low_confidence_keeps_current_value():
    n, claim, _ = await _run("on", _calibrated(verification_target=0.05), _result(vt_conf=0.6))
    assert claim["verification_target"] == "reality" and n == 0


@pytest.mark.asyncio
async def test_refused_field_keeps_current_value_others_still_apply():
    res = _result()
    res.answers["importance"] = DecisionAnswer(name="importance", type="refusal")
    n, claim, _ = await _run("on", _calibrated(verification_target=0.05, importance=0.3), res)
    assert claim["verification_target"] == "none"   # answered field applied
    assert claim["importance"] == 0.4               # refused field unchanged
    assert n == 1


def test_default_calibration_fits_only_fields_that_beat_call_b():
    from spectrue_core.runtime_config import CalibrationPolicyConfig
    s = CalibrationPolicyConfig().decision_sensors
    for f in ("verification_target", "claim_role", "harm", "importance"):
        assert s[f"meta_{f}"].reference_error_rate is not None and s[f"meta_{f}"].weight > 0
    for f in ("check_oracle", "time_sensitive", "satire"):
        assert s[f"meta_{f}"].weight == 0 and s[f"meta_{f}"].reference_error_rate is None
