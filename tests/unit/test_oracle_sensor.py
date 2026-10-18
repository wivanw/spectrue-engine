# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T025: fact-check DB matching from the Decisions sensor. No network."""

from __future__ import annotations

from types import SimpleNamespace

from spectrue_core.adapters.llm.oracle_sensor import oracle_questions
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.use_cases.verification.oracle_sensor import oracle_result_from_reading

CANDS = [{"claim_text": "Netflix CEO arrested", "rating": "False", "publisher": "AFP", "title": "No arrest"},
         {"claim_text": "Other", "rating": "True", "publisher": "X", "title": "y"}]


def _r(**a):
    return SimpleNamespace(result=DecisionResult(answers=a, model="m", usage={}), applied=True, latency_ms=1)


def test_questions_in_every_locale_with_candidates_as_choices():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        qs = oracle_questions(lang, CANDS)
        best = qs[0].payload
        assert [c["value"] for c in best["choices"]] == ["0", "1", "none"]
        assert "Prompt key" not in str([q.payload for q in qs])


def test_refuted_jackpot_and_miss():
    r = _r(best=DecisionAnswer(name="best", type="choice", choice="0", confidence=0.95),
           match=DecisionAnswer(name="match", type="score", score=3.8, confidence=0.9),
           status=DecisionAnswer(name="status", type="choice", choice="refuted", confidence=0.9),
           rating=DecisionAnswer(name="rating", type="score", score=0.2, confidence=0.9))
    out = oracle_result_from_reading(r, 2, jackpot_threshold=0.9)
    assert out["best_index"] == 0 and out["status"] == "REFUTED" and out["is_jackpot"] is True
    assert out["verified_score"] == 0.05 and out["danger_score"] == -1.0
    miss = oracle_result_from_reading(_r(best=DecisionAnswer(name="best", type="choice", choice="none", confidence=0.9)), 2,
                                      jackpot_threshold=0.9)
    assert miss["best_index"] == -1 and miss["status"] == "EMPTY"
