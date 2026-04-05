# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# This file is licensed under the GNU Affero General Public License.

from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE, DEEP_V2_MODE
from spectrue_core.pipeline.steps.retrieval.build_cluster_queries import BuildClusterQueriesStep
from spectrue_core.pipeline.steps.retrieval.build_queries import BuildQueriesStep
from spectrue_core.pipeline.steps.retrieval import cluster_web_search as cluster_web_search_module
from spectrue_core.pipeline.steps.retrieval.cluster_web_search import ClusterWebSearchStep
from spectrue_core.pipeline.steps.retrieval.web_search import WebSearchStep
from spectrue_core.runtime_config import DeepV2Config


@pytest.mark.asyncio
async def test_deep_v2_clustered_retrieval_reduces_search_calls():
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.deep_v2 = DeepV2Config()

    search_mgr = MagicMock()
    call_state = {"count": 0}

    async def _search_phase(query, max_results=5, depth="basic", topic="general", **kwargs):
        call_state["count"] += 1
        idx = call_state["count"]
        return None, [
            {
                "url": f"https://reuters.com/{idx}",
                "title": f"Result {idx}",
                "snippet": f"Snippet {idx}",
                "quote": "Sufficient evidence quote",
                "score": 0.9,
            }
        ]

    async def _fetch_batch(urls, stage=None):
        return {url: f"Content for {url}" for url in urls}

    search_mgr.search_phase = AsyncMock(side_effect=_search_phase)
    search_mgr.fetch_urls_content_batch = AsyncMock(side_effect=_fetch_batch)

    claims = [
        {"id": "c1", "text": "Claim one about X"},
        {"id": "c2", "text": "Claim two about X"},
        {"id": "c3", "text": "Claim three about Y"},
    ]

    ctx_deep = PipelineContext(mode=DEEP_MODE, claims=claims, lang="en")
    ctx_deep = ctx_deep.set_extra("prepared_fact", "Some text about X and Y.")
    ctx_deep = ctx_deep.set_extra("input_text", "Some text about X and Y.")
    ctx_deep = ctx_deep.set_extra("target_claims", claims)

    ctx_deep = await BuildQueriesStep().run(ctx_deep)
    await WebSearchStep(config=config, search_mgr=search_mgr, agent=MagicMock()).run(ctx_deep)

    deep_search_calls = search_mgr.search_phase.call_count
    deep_extract_calls = search_mgr.fetch_urls_content_batch.call_count

    search_mgr.search_phase.reset_mock()
    search_mgr.fetch_urls_content_batch.reset_mock()
    call_state["count"] = 0

    ctx_v2 = PipelineContext(mode=DEEP_V2_MODE, claims=claims, lang="en")
    ctx_v2 = ctx_v2.set_extra("prepared_fact", "Some text about X and Y.")
    ctx_v2 = ctx_v2.set_extra("input_text", "Some text about X and Y.")
    ctx_v2 = ctx_v2.set_extra(
        "cluster_representatives",
        {
            "cluster_a": [claims[0], claims[1]],
            "cluster_b": [claims[2]],
        },
    )
    ctx_v2 = ctx_v2.set_extra(
        "cluster_claims",
        {
            "cluster_a": [claims[0], claims[1]],
            "cluster_b": [claims[2]],
        },
    )

    ctx_v2 = await BuildClusterQueriesStep().run(ctx_v2)
    await ClusterWebSearchStep(config=config, search_mgr=search_mgr).run(ctx_v2)

    deep_v2_search_calls = search_mgr.search_phase.call_count
    deep_v2_extract_calls = search_mgr.fetch_urls_content_batch.call_count

    assert deep_v2_search_calls < deep_search_calls
    assert deep_v2_extract_calls <= deep_extract_calls


@pytest.mark.asyncio
async def test_deep_v2_clustered_retrieval_no_state_bleed_between_disjoint_entities():
    """
    Integration regression (US1): Ensure that disjoint entities in distinct clusters 
    do not bleed query state or retrieved evidence across clusters.
    """
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.deep_v2 = DeepV2Config()

    search_mgr = MagicMock()

    async def _search_phase(query, max_results=5, depth="basic", topic="general", **kwargs):
        if "apple" in query.lower():
            return None, [
                {
                    "url": "https://example.com/apple",
                    "title": "Apple news",
                    "snippet": "Apple is a fruit.",
                    "score": 0.9,
                }
            ]
        elif "banana" in query.lower():
            return None, [
                {
                    "url": "https://example.com/banana",
                    "title": "Banana news",
                    "snippet": "Banana is yellow.",
                    "score": 0.9,
                }
            ]
        return None, []

    async def _fetch_batch(urls, stage=None):
        out = {}
        for u in urls:
            if "apple" in u:
                out[u] = "Full Apple Content"
            elif "banana" in u:
                out[u] = "Full Banana Content"
        return out

    search_mgr.search_phase = AsyncMock(side_effect=_search_phase)
    search_mgr.fetch_urls_content_batch = AsyncMock(side_effect=_fetch_batch)

    claims = [
        {"id": "c_apple", "text": "Apple is a fruit"},
        {"id": "c_banana", "text": "Banana is yellow"},
    ]

    ctx_v2 = PipelineContext(mode=DEEP_V2_MODE, claims=claims, lang="en")
    ctx_v2 = ctx_v2.set_extra("target_claims", claims)
    # Simulate completely disjoint clusters
    ctx_v2 = ctx_v2.set_extra(
        "cluster_representatives",
        {
            "cluster_apple": [claims[0]],
            "cluster_banana": [claims[1]],
        },
    )
    ctx_v2 = ctx_v2.set_extra(
        "cluster_search_plans",
        [
            {"cluster_id": "cluster_apple", "search_queries": ["apple details"], "claims": [claims[0]]},
            {"cluster_id": "cluster_banana", "search_queries": ["banana facts"], "claims": [claims[1]]},
        ]
    )

    ctx_v2 = await ClusterWebSearchStep(config=config, search_mgr=search_mgr).run(ctx_v2)

    cluster_evidence_docs = ctx_v2.get_extra("cluster_evidence_docs", {})
    
    assert "cluster_apple" in cluster_evidence_docs
    assert "cluster_banana" in cluster_evidence_docs
    
    apple_ev = cluster_evidence_docs["cluster_apple"]
    banana_ev = cluster_evidence_docs["cluster_banana"]

    assert len(apple_ev) > 0, "Expected Apple evidence"
    assert len(banana_ev) > 0, "Expected Banana evidence"

    for ev in apple_ev:
        assert "apple" in ev["canonical_url"].lower()
        assert "banana" not in ev["canonical_url"].lower(), "State bleed! Banana evidence found in Apple cluster"

    for ev in banana_ev:
        assert "banana" in ev["canonical_url"].lower()
        assert "apple" not in ev["canonical_url"].lower(), "State bleed! Apple evidence found in Banana cluster"


@pytest.mark.asyncio
async def test_deep_v2_clustered_retrieval_caps_unique_urls_and_traces_phases(monkeypatch):
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.deep_v2 = DeepV2Config(max_unique_urls=2)

    search_mgr = MagicMock()
    search_map = {
        "q1": [
            {"url": "https://source-a.test/doc-1", "title": "A1", "snippet": "s", "score": 0.99},
            {"url": "https://source-a.test/doc-2", "title": "A2", "snippet": "s", "score": 0.60},
        ],
        "q2": [
            {"url": "https://source-b.test/doc-3", "title": "B1", "snippet": "s", "score": 0.95},
            {"url": "https://source-b.test/doc-4", "title": "B2", "snippet": "s", "score": 0.45},
        ],
        "q3": [
            {"url": "https://source-c.test/doc-5", "title": "C1", "snippet": "s", "score": 0.70},
            {"url": "https://source-c.test/doc-6", "title": "C2", "snippet": "s", "score": 0.40},
        ],
    }

    async def _search_phase(query, max_results=5, depth="basic", topic="general", **kwargs):
        return None, list(search_map.get(query, []))

    async def _fetch_batch(urls, stage=None):
        return {url: f"Content for {url}" for url in urls}

    search_mgr.search_phase = AsyncMock(side_effect=_search_phase)
    search_mgr.fetch_urls_content_batch = AsyncMock(side_effect=_fetch_batch)

    captured_events: list[str] = []

    def _capture_event(event: str, data=None):
        captured_events.append(event)

    monkeypatch.setattr(cluster_web_search_module.Trace, "event", staticmethod(_capture_event))

    claims = [{"id": "c1", "text": "Claim A"}]
    ctx_v2 = PipelineContext(mode=DEEP_V2_MODE, claims=claims, lang="en")
    ctx_v2 = ctx_v2.set_extra(
        "cluster_search_plans",
        [
            {
                "cluster_id": "cluster_1",
                "search_queries": ["q1", "q2", "q3"],
                "claims": claims,
            }
        ],
    )

    out_ctx = await ClusterWebSearchStep(config=config, search_mgr=search_mgr).run(ctx_v2)

    called_urls = list(search_mgr.fetch_urls_content_batch.await_args.args[0])
    assert len(called_urls) == 2
    assert set(called_urls) == {
        "https://source-a.test/doc-1",
        "https://source-b.test/doc-3",
    }

    trace_meta = out_ctx.get_extra("retrieval_search_trace")
    assert trace_meta["urls_before_cap"] == 6
    assert trace_meta["urls_total"] == 2
    assert trace_meta["url_cap"] == 2

    assert "retrieval.cluster_search.phase.search.start" in captured_events
    assert "retrieval.cluster_search.phase.search.complete" in captured_events
    assert "retrieval.cluster_search.phase.extract.start" in captured_events
    assert "retrieval.cluster_search.phase.extract.complete" in captured_events
    assert "retrieval.cluster_search.url_cap.applied" in captured_events
