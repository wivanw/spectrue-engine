# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for the OpenAI Decisions API client path (M145 R010–R013). No network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from spectrue_core.agents.llm_router import LLMRouter
from spectrue_core.llm.decisions import DecisionQuestion, parse_decision_response
from spectrue_core.llm.errors import LLMCallError, LLMFailureKind
from spectrue_core.llm.llm_client import LLMClient
from spectrue_core.runtime_config import EngineRuntimeConfig


QUESTIONS = [
    DecisionQuestion.choice("match_type", "How does the snippet relate?", {
        "EXACT": "explicit", "TOPIC": "same topic", "UNRELATED": "different",
    }),
    DecisionQuestion.predicate("is_primary", "Is this a primary source?"),
]

OK_BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {
            "type": "choice", "name": "match_type", "choice": "EXACT",
            "probabilities": [
                {"value": "EXACT", "probability": 0.95},
                {"value": "TOPIC", "probability": 0.04},
                {"value": "UNRELATED", "probability": 0.01},
            ],
            "confidence": 0.93,
        },
        {"type": "predicate", "name": "is_primary", "probability": 0.2},
    ],
    "usage": {"input_tokens": 177, "output_tokens": 0},
}


def _client(post: AsyncMock, meter: MagicMock | None = None) -> LLMClient:
    c = LLMClient(openai_api_key="sk-test", max_retries=2, meter=meter)
    c.client.post = post  # type: ignore[method-assign]
    return c


def _status_error(cls, status: int, msg: str):
    req = httpx.Request("POST", "https://api.openai.com/v1/decisions")
    return cls(msg, response=httpx.Response(status, request=req), body=None)


def test_parse_ok():
    res = parse_decision_response(OK_BODY, QUESTIONS)
    mt = res.get("match_type")
    assert mt.choice == "EXACT"
    assert mt.probabilities["TOPIC"] == pytest.approx(0.04)
    assert mt.top_confidence == pytest.approx(0.93)
    assert res.get("is_primary").top_confidence == pytest.approx(0.8)


def test_parse_refusal_is_not_a_label():
    body = {"answers": [
        {"type": "refusal", "name": "match_type"},
        {"type": "predicate", "name": "is_primary", "probability": 0.9},
    ]}
    res = parse_decision_response(body, QUESTIONS)
    a = res.get("match_type")
    assert a.is_refusal and a.choice is None and a.top_confidence is None


@pytest.mark.parametrize("body", [
    {},
    {"answers": [OK_BODY["answers"][0]]},  # missing is_primary
    {"answers": [OK_BODY["answers"][0], {"type": "predicate", "name": "is_primary", "probability": 1.7}]},
    {"answers": [OK_BODY["answers"][0], {"type": "choice", "name": "is_primary", "choice": "x"}]},
])
def test_parse_rejects_malformed(body):
    with pytest.raises(LLMCallError) as ei:
        parse_decision_response(body, QUESTIONS)
    assert ei.value.kind == LLMFailureKind.SCHEMA_VALIDATION_FAILED


@pytest.mark.asyncio
async def test_call_decision_posts_and_meters_input_only():
    post = AsyncMock(return_value=OK_BODY)
    meter = MagicMock()
    meter.record_completion.return_value = MagicMock(cost_credits=0.001)
    res = await _client(post, meter).call_decision(
        model="gpt-6-luna", input="claim + snippet", questions=QUESTIONS, trace_kind="t"
    )
    assert res.get("match_type").choice == "EXACT"
    path = post.call_args.args[0]
    assert path == "/decisions"
    sent = post.call_args.kwargs["body"]
    assert sent["model"] == "gpt-6-luna" and len(sent["questions"]) == 2
    kw = meter.record_completion.call_args.kwargs
    assert kw["model"] == "gpt-6-luna-decisions"
    assert kw["usage"]["input_tokens"] == 177 and kw["usage"]["output_tokens"] == 0


@pytest.mark.asyncio
async def test_call_decision_rate_limit_retries_then_succeeds():
    post = AsyncMock(side_effect=[_status_error(openai.RateLimitError, 429, "slow down"), OK_BODY])
    c = _client(post)
    c_sleep = AsyncMock()
    import spectrue_core.llm.llm_client as mod
    orig = mod.asyncio.sleep
    mod.asyncio.sleep = c_sleep
    try:
        res = await c.call_decision(model="gpt-6-luna", input="x", questions=QUESTIONS)
    finally:
        mod.asyncio.sleep = orig
    assert res.get("is_primary").probability == pytest.approx(0.2)
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_call_decision_too_long_is_typed_and_not_retried():
    post = AsyncMock(side_effect=_status_error(openai.BadRequestError, 400, "input exceeds maximum context length"))
    with pytest.raises(LLMCallError) as ei:
        await _client(post).call_decision(model="gpt-6-luna", input="x", questions=QUESTIONS)
    assert ei.value.kind == LLMFailureKind.INPUT_TOO_LONG
    assert post.await_count == 1


@pytest.mark.asyncio
async def test_call_decision_schema_error_not_retried():
    post = AsyncMock(return_value={"answers": []})
    with pytest.raises(LLMCallError) as ei:
        await _client(post).call_decision(model="gpt-6-luna", input="x", questions=QUESTIONS)
    assert ei.value.kind == LLMFailureKind.SCHEMA_VALIDATION_FAILED
    assert post.await_count == 1


@pytest.mark.asyncio
async def test_chat_completions_endpoint_rejects_decisions():
    c = LLMClient(base_url="https://api.deepseek.com")
    with pytest.raises(LLMCallError) as ei:
        await c.call_decision(model="deepseek-v4-pro", input="x", questions=QUESTIONS)
    assert ei.value.kind == LLMFailureKind.UNSUPPORTED_ROUTE


@pytest.mark.asyncio
async def test_router_routes_decision_to_openai_client():
    openai_client = MagicMock(spec=LLMClient)
    openai_client.call_decision = AsyncMock(return_value="ok")
    chat_client = MagicMock(spec=LLMClient)
    router = LLMRouter(openai_client=openai_client, chat_client=chat_client, chat_model_names=["deepseek-v4-pro"])
    assert await router.call_decision(model="gpt-6-luna", input="x", questions=QUESTIONS) == "ok"
    chat_client.call_decision.assert_not_called()


def test_decisions_config_defaults_off(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("DECISIONS_") or k == "MODEL_DECISIONS":
            monkeypatch.delenv(k, raising=False)
    cfg = EngineRuntimeConfig.load_from_env().decisions
    assert cfg.model == "gpt-6-luna"
    assert all(cfg.mode(g) == "off" for g in cfg.gate_modes)


def test_decisions_config_env(monkeypatch):
    monkeypatch.setenv("DECISIONS_GATE_SEARCH_RELEVANCE", "shadow")
    cfg = EngineRuntimeConfig.load_from_env().decisions
    assert cfg.mode("search_relevance") == "shadow"
    assert cfg.mode("extract") == "off"


def test_decisions_config_invalid_mode_raises(monkeypatch):
    # Constitution III: a misconfigured gate must not silently degrade to "off".
    monkeypatch.setenv("DECISIONS_GATE_EXTRACT", "bogus")
    with pytest.raises(ValueError):
        EngineRuntimeConfig.load_from_env()


def test_sensor_calibration_defaults_uncalibrated_except_fitted_metadata():
    sensors = EngineRuntimeConfig.load_from_env().calibration.decision_sensors
    fitted = {"meta_verification_target", "meta_claim_role", "meta_harm", "meta_importance"}
    assert all(c.weight == 0.0 and c.bias == 0.0 and c.reference_error_rate is None
               for n, c in sensors.items() if n not in fitted)
