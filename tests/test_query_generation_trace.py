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

    # First call returns empty -> triggers retry prompt; second call returns valid JSON.
    responses = [
        "",
        json.dumps(
            {
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
        ),
    ]

    async def fake_create(*args, **kwargs):
        if not responses:
            return _FakeResponse("")
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)

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

    # Inspect trace file for prompt sanitization and retry consistency.
    trace_file = _tmp_trace_dir() / f"{trace_id}.jsonl"
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    prompt_events = [e for e in events if e["event"] == "llm.prompt"]

    kinds = {e["data"]["kind"] for e in prompt_events}
    assert {"query_generation", "query_generation.retry"} <= kinds

    for ev in prompt_events:
        prompt_payload = ev["data"]["prompt"]
        assert isinstance(prompt_payload, dict)
        assert prompt_payload.get("len") and prompt_payload.get("sha256")
        assert "TAIL_MARKER" in prompt_payload.get("tail", "")

    # Ensure no legacy rewrite reason remains.
    assert not any("long_fact_default" in json.dumps(e) for e in events)
