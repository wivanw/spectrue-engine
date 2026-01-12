import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.pipeline.steps.retrieval.build_queries import BuildQueriesStep
from spectrue_core.pipeline.steps.retrieval.web_search import WebSearchStep
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_MODE
from spectrue_core.utils.trace import Trace

@pytest.mark.asyncio
async def test_cegs_trace_flow():
    # Setup Context
    claims = [
        {"id": "c1", "text": "Test claim 1", "subject": "TestSubject", "context_entities": ["ctx1"], "subject_entities": ["subj1"]},
        {"id": "c2", "text": "Test claim 2", "subject": "TestSubject", "context_entities": ["ctx1"], "subject_entities": ["subj2"]}
    ]
    ctx = PipelineContext(
        mode=DEEP_MODE,
        claims=claims,
        lang="en",
        extras={"target_claims": claims, "prepared_fact": "Test document text about TestSubject and ctx1."}
    )
    
    # Mock Deps
    search_mgr = MagicMock()
    search_mgr.search_phase = AsyncMock(return_value=("snippet", [{"url": "u1", "title": "T", "content": "C", "score": 0.9}]))
    search_mgr.fetch_url_content = AsyncMock(return_value="Full content")
    
    agent = MagicMock()
    config = MagicMock()
    
    # Trace Spy
    with patch.object(Trace, 'event') as trace_spy:
        # 1. Build Queries
        build_step = BuildQueriesStep()
        ctx = await build_step.run(ctx)
        
        # Verify Plan Trace
        # retrieval.doc_plan should be emitted by cegs_mvp logic called by BuildQueriesStep
        # Check args of trace_spy
        found_plan = False
        for call in trace_spy.call_args_list:
            if call[0][0] == "retrieval.doc_plan":
                found_plan = True
                break
        assert found_plan, "retrieval.doc_plan not found"
        
        # 2. Web Search
        search_step = WebSearchStep(config, search_mgr, agent)
        ctx = await search_step.run(ctx)
        
        # Verify Search/Cluster/Match Traces
        expected_events = [
            "retrieval.doc.search",
            "retrieval.doc.clusters",
            "retrieval.doc.extract",
            "retrieval.pool.match",
            "retrieval.claim.deficit"
        ]
        
        emitted_events = [call[0][0] for call in trace_spy.call_args_list]
        for evt in expected_events:
            assert evt in emitted_events, f"Event {evt} not found in {emitted_events}"
            
        # Verify Escalation (if deficit triggered)
        # In our mock, search returns good score, so likely no deficit.
        # Let's force deficit?
        # If score 0.9 and overlap -> match found -> no deficit -> no escalation.
        
        assert "retrieval.pool.match" in emitted_events
