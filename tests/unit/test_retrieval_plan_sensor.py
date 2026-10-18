# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T022: call-B retrieval plan from call A + the Decisions sensor. No network."""

from __future__ import annotations

from types import SimpleNamespace

from spectrue_core.adapters.llm.retrieval_plan_sensor import plan_questions
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.use_cases.claims.retrieval_plan import compare_plans, plan_from_reading


def _reading(**answers):
    return SimpleNamespace(result=DecisionResult(answers=answers, model="m", usage={}), applied=True, latency_ms=1)


def _choice(name, v, probs=None, conf=0.9):
    return DecisionAnswer(name=name, type="choice", choice=v, probabilities=probs or {v: conf}, confidence=conf)


def test_plan_questions_cover_metadata_and_call_b_fields_in_every_locale():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        names = [q.name for q in plan_questions(lang)]
        assert {"verification_target", "claim_role", "harm", "importance", "satire",
                "evidence_need", "search_method", "needs_primary", "structure_type"} <= set(names)
        assert "Prompt key" not in str([q.payload for q in plan_questions(lang)])


def test_plan_from_reading_derives_call_b_fields():
    r = _reading(
        verification_target=_choice("verification_target", "reality", {"reality": 0.9, "none": 0.04}),
        claim_role=_choice("claim_role", "core"),
        harm=DecisionAnswer(name="harm", type="score", score=2.0, confidence=0.7),
        importance=DecisionAnswer(name="importance", type="score", score=3.0, confidence=0.8),
        satire=DecisionAnswer(name="satire", type="predicate", probability=0.02, confidence=0.98),
        evidence_need=_choice("evidence_need", "news_report"),
        search_method=_choice("search_method", "news"),
        needs_primary=DecisionAnswer(name="needs_primary", type="predicate", probability=0.7, confidence=0.7),
        structure_type=_choice("structure_type", "event"),
    )
    p = plan_from_reading(r, normalized_text="VK reported a hit.", search_queries=["VK data center hit"], lang="uk")
    assert p["verification_target"] == "reality" and p["check_worthiness"] == 0.96 and p["check_oracle"] is True
    assert (p["harm_potential"], p["importance"], p["claim_category"]) == (3, 0.75, "FACTUAL")
    assert p["evidence_req"] == {"needs_primary": True, "needs_2_independent": False}
    assert p["structure"]["type"] == "event" and p["search_queries"] == ["VK data center hit"]
    assert p["search_locale_plan"] == {"primary": "uk", "fallback": ["en"]} and p["metadata_confidence"] == "medium"


def test_satire_and_opinion_categories_and_refused_fields():
    sat = plan_from_reading(_reading(satire=DecisionAnswer(name="satire", type="predicate", probability=0.9, confidence=0.9)),
                            normalized_text="x", search_queries=[], lang="en")
    assert sat["claim_category"] == "SATIRE" and "verification_target" not in sat   # refused field left to defaults
    op = plan_from_reading(_reading(verification_target=_choice("verification_target", "none", {"none": 0.95})),
                           normalized_text="x", search_queries=[], lang="en")
    assert op["claim_category"] == "OPINION" and op["check_oracle"] is False and op["check_worthiness"] == 0.05


def test_compare_plans_reports_both_sides():
    c = compare_plans({"claim_role": "thesis"}, {"claim_role": "core"})
    assert c["claim_role"] == {"call_b": "thesis", "sensor": "core"}
