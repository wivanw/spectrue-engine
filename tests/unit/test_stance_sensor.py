# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T020: Decisions stance sensor. No network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.stance_sensor import STANCES, stance_questions
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.runtime_config import DecisionsConfig, uncalibrated_sensors
from spectrue_core.use_cases.evidence.stance_sensor import apply_stance_sensor, row_text

CLAIMS = {"c1": "VK reported a hit on a data center."}


def _rows():
    return [{"claim_id": "c1", "url": "https://a/vk", "stance": "context", "quote": "VK said its data center was hit."},
            {"claim_id": "c1", "url": "https://a/none", "stance": "context"},            # no text: not read
            {"claim_id": "c9", "url": "https://a/other", "stance": "context", "quote": "x"}]  # unknown claim


def _result(choice="support", conf=0.95):
    probs = {s: (conf if s == choice else (1 - conf) / 3) for s in STANCES}
    return DecisionResult(answers={"stance": DecisionAnswer(name="stance", type="choice", choice=choice,
                                                            probabilities=probs, confidence=conf)},
                          model="m", usage={})


async def _run(mode, sensors, result):
    client = MagicMock(); client.call_decision = AsyncMock(return_value=result)
    rows = _rows()
    with patch("spectrue_core.use_cases.evidence.stance_sensor.Trace") as tr, \
         patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        n = await apply_stance_sensor(client, DecisionsConfig(gate_modes={"stance": mode}), sensors, rows, CLAIMS, lang="uk")
    return n, rows, client, tr


def test_questions_cover_every_stance_in_every_locale():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        q = stance_questions(lang)[0].payload
        assert [c["value"] for c in q["choices"]] == list(STANCES)
        assert all(c["description"] and "Prompt key" not in c["description"] for c in q["choices"])


def test_row_text_prefers_quote_then_content():
    assert row_text({"quote": " q ", "content": "c"}) == "q"
    assert row_text({"content_excerpt": "e"}) == "e"


@pytest.mark.asyncio
async def test_off_makes_no_call():
    n, rows, client, _ = await _run("off", uncalibrated_sensors(), _result())
    assert n == 0 and not client.call_decision.called


@pytest.mark.asyncio
async def test_shadow_reads_only_rows_with_claim_and_text_and_changes_nothing():
    n, rows, client, tr = await _run("shadow", uncalibrated_sensors(), _result())
    assert client.call_decision.call_count == 1 and n == 0 and rows[0]["stance"] == "context"
    summary = [c.args[1] for c in tr.event.call_args_list if c.args[0] == "stance_sensor.summary"][0]
    assert summary == {"rows": 3, "read": 1, "agree": 0, "changed": 0}


@pytest.mark.asyncio
async def test_on_the_sensor_is_the_stance_classifier():
    n, rows, _, _ = await _run("on", uncalibrated_sensors(), _result("support", 0.6))
    assert n == 1 and rows[0]["stance"] == "support" and rows[0]["stance_matrix"] == "context"
    n, rows, _, _ = await _run("on", uncalibrated_sensors(), _result("context", 0.9))   # agrees: nothing to change
    assert n == 0 and "stance_matrix" not in rows[0]
