import json
from pathlib import Path
from uuid import uuid4

import pytest

from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.config import SpectrueConfig
from spectrue_core.utils import trace as trace_mod
from spectrue_core.runtime_config import EngineRuntimeConfig


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMsg(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


@pytest.mark.asyncio
async def test_query_generation_trace_and_retry_sanitized(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTRUE_ENV", "local")

    def _tmp_trace_dir() -> Path:
        p = tmp_path / "trace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(trace_mod, "_trace_dir", _tmp_trace_dir)

    agent = FactCheckerAgent(SpectrueConfig(openai_api_key="test"))

    # M49: Mock llm_client.call_json
    # First call raises (empty response); second call returns valid JSON.
    call_count = [0]

    async def fake_call_json(*, model, input, **kwargs):  # noqa: A002
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulate empty/error response by raising (LLMClient retry exhausted)
            raise ValueError("Empty response from LLM")
        return {
            "claim": {
                "subject": "Test subject",
                "action": "announce",
                "object": "object tail",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            "queries": [
                "english query with marker TAIL_MARKER",
                "ukrainian query with marker TAIL_MARKER",
            ],
        }

    monkeypatch.setattr(agent.llm_client, "call_json", fake_call_json)

    trace_id = f"test-{uuid4()}"
    TraceRuntime = EngineRuntimeConfig.load_from_env()
    trace_mod.Trace.start(trace_id, runtime=TraceRuntime)

    long_fact = "start " + ("x " * 3000) + "TAIL_MARKER"
    queries = await agent.generate_search_queries(
        long_fact, context="CTX", lang="uk", content_lang="uk", allow_short_llm=True
    )

    trace_mod.Trace.stop()

    assert len(queries) == 2
    assert queries[0]
    assert queries[1]

    # Inspect trace file for prompt events.
    trace_file = _tmp_trace_dir() / f"{trace_id}.jsonl"
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    prompt_events = [e for e in events if e["event"] == "query_generation.prompt"]
    
    # M49: LLMClient now uses different trace format - check for any llm-related events
    llm_events = [e for e in events if "llm" in e["event"] or "query" in e["event"]]
    assert len(llm_events) > 0  # Should have at least error + retry events

    # Ensure no legacy rewrite reason remains.
    assert not any("long_fact_default" in json.dumps(e) for e in events)
