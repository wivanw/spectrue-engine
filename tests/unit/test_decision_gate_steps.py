# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Thin G1/G2 steps and the G5 agent path (M145). No network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.decision_gates import SensorReading
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.pipeline.contracts import RAW_SEARCH_RESULTS_KEY, RawSearchResult, RawSearchResults
from spectrue_core.pipeline.steps.evaluate_semantic_gating import EvaluateSemanticGatingStep
from spectrue_core.pipeline.steps.retrieval.search_relevance_gate import SearchRelevanceGateStep
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors

CLAIMS = [{"id": "c1", "text": "a", "check_worthiness": 0.8}]


class Ctx:
    def __init__(self, extras):
        self._e, self.claims, self.lang = extras, CLAIMS, "en"

    def get_extra(self, k, d=None):
        return self._e.get(k, d)

    def set_extra(self, k, v):
        return Ctx({**self._e, k: v})


def _reading(applied=True, **p):
    return SensorReading(result=DecisionResult(
        answers={k: DecisionAnswer(name=k, type="predicate", probability=v) for k, v in p.items()},
        model="m", usage={}), applied=applied, latency_ms=1)


def _cal(**over):
    s = uncalibrated_sensors()
    for k, ref in over.items():
        s[k] = DecisionSensorCalibration(version="t", weight=1.0, reference_error_rate=ref)
    return s


@pytest.mark.asyncio
async def test_g2_step_off_is_noop():
    ctx = Ctx({"eligible_claims": CLAIMS})
    assert await EvaluateSemanticGatingStep(agent=None).run(ctx) is ctx


@pytest.mark.asyncio
async def test_g2_step_on_updates_eligible_claims():
    agent = SimpleNamespace(relevance_skill=SimpleNamespace(read_verifiability=AsyncMock(return_value=_reading(checkable=0.5))))
    step = EvaluateSemanticGatingStep(agent=agent, decisions=DecisionsConfig(gate_modes={"verifiability": "on"}),
                                      sensors=_cal(verifiability=None))
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        out = await step.run(Ctx({"eligible_claims": CLAIMS}))
    assert out.get_extra("eligible_claims")[0]["check_worthiness"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_g1_step_writes_posterior_similarity_for_rerank():
    raw = RawSearchResults(plan_id="p", results=(RawSearchResult(
        plan_id="p", query_id="q", query="q", claim_id="c1",
        provider_payload=[{"url": "u1", "similarity_score": 0.5, "decision_evidence": {"c1": 0.9}}]),))
    step = SearchRelevanceGateStep(agent=SimpleNamespace(relevance_skill=None),
                                   decisions=DecisionsConfig(gate_modes={"search_relevance": "on"}),
                                   sensors=_cal(search_relevance=None))
    with patch("spectrue_core.use_cases.retrieval.decision_gates.Trace"):
        out = await step.run(Ctx({RAW_SEARCH_RESULTS_KEY: raw, "eligible_claims": CLAIMS}))
    src = out.get_extra(RAW_SEARCH_RESULTS_KEY).results[0].provider_payload[0]
    assert src["similarity_score"] == pytest.approx(0.9)


def _agent(mode, sensors, *, decision=None, llm_json=None):
    a = FactCheckerAgent.__new__(FactCheckerAgent)
    a.config = None
    a.runtime = SimpleNamespace(
        llm=SimpleNamespace(enable_inline_source_verification=True, model_inline_source_verification="m"),
        decisions=DecisionsConfig(gate_modes={"inline_source": mode}),
        calibration=SimpleNamespace(decision_sensors=sensors),
    )
    a.llm_client = MagicMock()
    a.llm_client.call_decision = AsyncMock(return_value=decision)
    a.llm_client.call_json = AsyncMock(return_value=llm_json)
    return a


def _preds(rel, prim):
    return DecisionResult(answers={
        "is_relevant": DecisionAnswer(name="is_relevant", type="predicate", probability=rel),
        "is_primary": DecisionAnswer(name="is_primary", type="predicate", probability=prim)},
        model="m", usage={"input_tokens": 5})


SRC = {"url": "https://gov.example/s", "domain": "gov.example", "title": "t"}
LLM = {"is_relevant": True, "is_primary": False, "reason": "coverage"}


@pytest.mark.asyncio
async def test_g5_off_uses_llm():
    a = _agent("off", uncalibrated_sensors(), llm_json=LLM)
    assert await a.verify_inline_source_relevance([{"text": "c"}], SRC) == LLM
    a.llm_client.call_decision.assert_not_called()


@pytest.mark.asyncio
async def test_g5_on_calibrated_within_error_replaces_llm():
    a = _agent("on", _cal(inline_relevant=0.05, inline_primary=0.05), decision=_preds(0.99, 0.99), llm_json=LLM)
    out = await a.verify_inline_source_relevance([{"text": "c"}], SRC)
    assert out["is_primary"] is True and out["reason"] == "decisions_api"
    a.llm_client.call_json.assert_not_called()


@pytest.mark.asyncio
async def test_g5_on_uncalibrated_keeps_llm():
    a = _agent("on", uncalibrated_sensors(), decision=_preds(0.99, 0.99), llm_json=LLM)
    assert await a.verify_inline_source_relevance([{"text": "c"}], SRC) == LLM
