# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import asyncio

from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent


def test_feature_flag_query_rewrite_short_default_false(monkeypatch):
    monkeypatch.delenv("SPECTRUE_LLM_QUERY_REWRITE_SHORT", raising=False)
    cfg = EngineRuntimeConfig.load_from_env()
    assert cfg.features.query_rewrite_short is False


def test_llm_concurrency_is_clamped(monkeypatch):
    monkeypatch.setenv("OPENAI_CONCURRENCY", "999")
    cfg = EngineRuntimeConfig.load_from_env()
    assert cfg.llm.concurrency == 16

    monkeypatch.setenv("OPENAI_CONCURRENCY", "0")
    cfg2 = EngineRuntimeConfig.load_from_env()
    assert cfg2.llm.concurrency == 1


def test_generate_search_queries_short_fact_does_not_call_llm(monkeypatch):
    runtime = EngineRuntimeConfig.load_from_env()
    config = SpectrueConfig(openai_api_key="test", tavily_api_key="test", google_fact_check_key="test", runtime=runtime)
    agent = FactCheckerAgent(config)

    async def _fail_call_json(**_kwargs):
        raise AssertionError("nano LLM should not be called for short facts by default")

    monkeypatch.setattr(agent.llm_client, "call_json", _fail_call_json)

    res = asyncio.run(agent.generate_search_queries("Short claim.", lang="en", content_lang="en"))
    assert isinstance(res, list)
    assert len(res) >= 1


def test_responses_api_defaults():
    cfg = EngineRuntimeConfig.load_from_env()
    assert cfg.llm.cluster_timeout_sec == 60.0
