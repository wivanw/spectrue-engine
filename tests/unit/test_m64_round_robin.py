
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.config import SpectrueConfig

@pytest.fixture
def mock_config():
    config = MagicMock(spec=SpectrueConfig)
    config.max_queries = 3
    config.runtime = MagicMock()
    config.runtime.search.tavily_max_results = 5
    config.tavily_api_key = "test-key"
    config.google_fact_check_key = "test-key"
    config.google_search_api_key = "test-key"
    config.google_search_cse_id = "test-cse-id"
    return config

@pytest.fixture
def pipeline(mock_config):
    search_mgr = MagicMock()
    # Explicitly mock async methods
    search_mgr.search_tier1 = AsyncMock()
    search_mgr.search_tier2 = AsyncMock()
    search_mgr.search_google_cse = AsyncMock()
    search_mgr.check_oracle_hybrid = AsyncMock()
    # Explicitly mock sync methods to avoid unawaited coroutine warnings
    search_mgr.reset_metrics = MagicMock()
    search_mgr.calculate_cost = MagicMock(return_value=0.0)
    search_mgr.get_search_meta = MagicMock(return_value={})
    search_mgr.tavily_calls = 0

    llm_client = AsyncMock()
    # Mocking cluster_evidence and score_evidence to avoid errors during full pipeline execution
    agent = AsyncMock()
    agent.cluster_evidence.return_value = []
    agent.score_evidence.return_value = {"verified_score": 0.5}
    
    p = ValidationPipeline(
        config=mock_config,
        agent=agent
    )
    # Inject mocked search_mgr manually as constructor creates its own
    p.search_mgr = search_mgr
    # Patch agent explicitly if passed in init differently
    p.agent = agent 
    return p

class TestM64RoundRobin:
    
    def test_round_robin_selection_basic(self, pipeline):
        """M64: Verify round-robin selection ensures topic coverage."""
        claims = [
            {
                "id": "c1", "topic_key": "Topic A", "type": "core", "importance": 1.0, "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "A1 Core", "role": "CORE", "score": 1.0},
                    {"text": "A2 Local", "role": "LOCAL", "score": 0.8}
                ]
            },
            {
                "id": "c2", "topic_key": "Topic B", "type": "core", "importance": 0.9, "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "B1 Core", "role": "CORE", "score": 1.0}
                ]
            }
        ]
        
        # Max 2 queries -> Should pick 1 from A (A1) and 1 from B (B1)
        # instead of 2 from A.
        queries = pipeline._select_diverse_queries(claims, max_queries=2)
        assert len(queries) == 2
        assert "A1 Core" in queries
        assert "B1 Core" in queries

    def test_round_robin_depth_fill(self, pipeline):
        """M64: Verify depth fill if budget allows (Round 2)."""
        claims = [
            {
                "id": "c1", "topic_key": "Topic A", "type": "core", "importance": 1.0, "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "A1 Core", "role": "CORE", "score": 1.0},
                    {"text": "A2 Local", "role": "LOCAL", "score": 0.8}
                ]
            },
            # Topic B has only 1 candidate
            {
                "id": "c2", "topic_key": "Topic B", "type": "core", "importance": 0.9, "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "B1 Core", "role": "CORE", "score": 1.0}
                ]
            }
        ]
        
        # Max 3 queries -> Round 1: A1, B1. Round 2: A2.
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        assert len(queries) == 3
        assert "A1 Core" in queries
        assert "B1 Core" in queries
        assert "A2 Local" in queries

    def test_fuzzy_deduplication(self, pipeline):
        """M64: Verify fuzzy deduplication removes similar queries."""
        claims = [
            {
                "id": "c1", "topic_key": "Topic A", "type": "core", "importance": 1.0, "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "one two three four five six seven eight nine ten", "role": "CORE", "score": 1.0},
                    # Very similar query (11th word added), should be Deduped (10/11 = 0.909 > 0.9)
                    {"text": "one two three four five six seven eight nine ten eleven", "role": "LOCAL", "score": 0.9} 
                ]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        assert len(queries) == 1
        assert queries[0] == "one two three four five six seven eight nine ten"

    @pytest.mark.asyncio
    async def test_tavily_topic_guardrail_news(self, pipeline):
        """M64/M65: Verify 'news' topic is passed to Unified Search for 'news' intent."""
        # Setup
        pipeline.agent.extract_claims = AsyncMock(return_value=(
            [], # claims
            False, # check_oracle
            "news" # article_intent -> triggers topic="news"
        ))
        pipeline._select_diverse_queries = MagicMock(return_value=["Query 1"])
        pipeline._can_add_search = MagicMock(return_value=True) # Allow search
        
        # Setup Oracle to MISS
        pipeline.search_mgr.check_oracle_hybrid = AsyncMock(return_value=None)
        
        # Setup Search Results
        pipeline.search_mgr.search_unified = AsyncMock(return_value=("", []))
    
        # Execute
        await pipeline.execute("Fake text", search_type="smart", gpt_model="gpt-5-nano", lang="en")
    
        # Verify Unified Called with topic="news"
        call_args = pipeline.search_mgr.search_unified.call_args
        assert call_args is not None
        # Args: query, topic, intent
        # Keyword args or positional?
        # definition: search_unified(self, query: str, topic: str = "general", intent: str = "news")
        # Call used kwargs: search_unified(primary_query, topic=tavily_topic, intent=article_intent)
        assert call_args.kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_tavily_topic_guardrail_general(self, pipeline):
        """M64/M65: Verify 'general' topic is passed for 'evergreen' intent."""
        # Setup
        pipeline.agent.extract_claims = AsyncMock(return_value=(
            [], False, "evergreen" # intent -> topic="general"
        ))
        pipeline._select_diverse_queries = MagicMock(return_value=["Query 1"])
        
        pipeline.search_mgr.search_unified = AsyncMock(return_value=("", []))
        pipeline.search_mgr.check_oracle_hybrid = AsyncMock(return_value=None)
    
        await pipeline.execute("Fake text", search_type="smart", gpt_model="gpt-5-nano", lang="en")
        
        # Verify Unified Called with topic="general"
        call_args = pipeline.search_mgr.search_unified.call_args
        assert call_args is not None
        assert call_args.kwargs.get("topic") == "general"
