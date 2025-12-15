import asyncio
import inspect
import json

from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.config import SpectrueConfig


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


def test_query_generation_prompt_has_no_probe_policy(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    captured = {"prompt": None}

    async def _fake_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        payload = {
            "claim": {
                "subject": "NASA",
                "action": "",
                "object": "new moon Earth",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            "queries": ["NASA new moon Earth official statement"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)

    res = asyncio.run(
        agent.generate_search_queries(
            "NASA reported a new moon.",
            lang="en",
            content_lang="en",
            allow_short_llm=True,
        )
    )
    assert isinstance(res, list)
    assert captured["prompt"] is not None
    assert "probe_policy" not in captured["prompt"]
    assert "\"claim\"" in captured["prompt"]
    assert "\"queries\"" in captured["prompt"]


def test_no_domain_sanitizer_layer_in_code():
    src = inspect.getsource(FactCheckerAgent)
    assert "_sanitize_domain_probes" not in src
    assert "event_markers" not in src
    assert "protest_markers" not in src
    assert "sport_markers" not in src


def test_prompt_contains_full_statement_tail_no_truncation(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    tail = "TAIL_SUBSTRING_XYZ_123"
    statement = ("A" * 2100) + tail
    context_tail = "CTX_TAIL_456"
    context = ("B" * 1200) + context_tail

    captured = {"prompt": None}

    async def _fake_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        payload = {
            "claim": {
                "subject": "NASA",
                "action": "",
                "object": "new moon Earth",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            # M44: Return 2 queries (EN, UK)
            "queries": ["NASA new moon Earth official statement update today", "NASA нова Місяць Земля офіційна заява"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)

    res = asyncio.run(agent.generate_search_queries(statement, context=context, lang="en", content_lang="en", allow_short_llm=True))
    # M44: Now returns 2 queries (EN, UK)
    assert len(res) == 2
    assert captured["prompt"] is not None
    assert tail in captured["prompt"]
    assert context_tail in captured["prompt"]


def test_prompt_has_no_domain_marker_instructions(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    captured = {"prompt": None}

    async def _fake_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        payload = {
            "claim": {
                "subject": "NASA",
                "action": "",
                "object": "new moon Earth",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            "queries": ["NASA new moon Earth official statement update today"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)
    asyncio.run(agent.generate_search_queries("NASA reported a new moon.", lang="en", content_lang="en", allow_short_llm=True))

    p = (captured["prompt"] or "").lower()
    assert "venue" not in p
    assert "organizers said" not in p
    assert "police statement" not in p
    assert "league statement" not in p


def test_security_removed_from_queries_when_not_in_statement(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    async def _fake_create(**_kwargs):
        payload = {
            "claim": {
                "subject": "Band",
                "action": "",
                "object": "flag incident",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            "queries": ["Band flag incident official statement update today security"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)

    statement = "At the concert the band stopped the show."
    res = asyncio.run(agent.generate_search_queries(statement, lang="en", content_lang="en", allow_short_llm=True))
    joined = " | ".join(res).lower()
    assert "security" not in joined


def test_exactly_one_probe_required_when_missing_fields(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    async def _fake_create(**_kwargs):
        payload = {
            "claim": {
                "subject": "NASA",
                "action": "",
                "object": "new moon Earth",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            # M44: Return 2 queries (EN, UK)
            "queries": ["NASA new moon Earth official statement", "NASA нова Місяць Земля офіційна заява"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)

    res = asyncio.run(agent.generate_search_queries("NASA reported a new moon.", lang="en", content_lang="en", allow_short_llm=True))
    # M44: Now returns 2 queries
    assert len(res) == 2
    q = res[0].lower()
    # No probe words like where/when in queries
    assert "where" not in q
    assert "when" not in q
    # Official statement should be in query
    assert "official statement" in q


def test_accepts_exactly_one_probe_when_missing_fields(monkeypatch):
    cfg = SpectrueConfig(openai_api_key="test")
    agent = FactCheckerAgent(cfg)

    async def _fake_create(**_kwargs):
        payload = {
            "claim": {
                "subject": "NASA",
                "action": "",
                "object": "new moon Earth",
                "where": None,
                "when": None,
                "by_whom": None,
            },
            # M44: Return 2 queries (EN, UK)
            "queries": ["NASA new moon Earth official statement", "NASA нова Місяць Земля офіційна заява"],
        }
        return _Resp(json.dumps(payload))

    monkeypatch.setattr(agent.client.chat.completions, "create", _fake_create)

    res = asyncio.run(agent.generate_search_queries("NASA reported a new moon.", lang="en", content_lang="en", allow_short_llm=True))
    # M44: Now returns 2 queries
    assert len(res) == 2
    assert "nasa" in res[0].lower()
    assert "new moon earth" in res[0].lower()
    assert res[0].lower().count("official statement") == 1
