
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline

@pytest.mark.unit
class TestDeepScoringPipeline:
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.extract_claims = AsyncMock()
        agent.score_evidence = AsyncMock()
        agent.score_evidence_parallel = AsyncMock()
        # Mock other required methods to avoid failures
        agent.cluster_evidence = AsyncMock(return_value=[])
        agent.verify_search_relevance = AsyncMock(return_value={"is_relevant": True})
        agent.verify_inline_source_relevance = AsyncMock(return_value={"is_relevant": True})
        return agent

    @pytest.fixture
    def mock_search_mgr(self):
        mgr = MagicMock()
        mgr.search_unified = AsyncMock(return_value=("Context", []))
        mgr.check_oracle_hybrid = AsyncMock(return_value=None)
        mgr.calculate_cost = MagicMock(return_value=10)
        mgr.can_afford = MagicMock(return_value=True)
        mgr.reset_metrics = MagicMock()
        mgr.apply_evidence_acquisition_ladder = AsyncMock(return_value=[])
        mgr.estimate_hop_cost = MagicMock(return_value=10)
        # Mock web_tool._tavily._meter for meter swap logic in pipeline
        mgr.web_tool = MagicMock()
        mgr.web_tool._tavily = MagicMock()
        mgr.web_tool._tavily._meter = MagicMock()
        mgr.tavily_calls = 0
        return mgr

    @pytest.fixture
    def pipeline(self, mock_config, mock_agent, mock_search_mgr):
        # Mock CalibrationRegistry to avoid reading from real config
        with patch("spectrue_core.verification.pipeline.CalibrationRegistry") as MockRegistry:
            mock_registry_instance = MagicMock()
            
            # Setup mock model with valid score return
            mock_model = MagicMock()
            mock_model.score.return_value = (0.5, {"mock_trace": True})
            
            # Configure get_model to return this mock model
            mock_registry_instance.get_model.return_value = mock_model
            
            # Also need policy for attribute access if any
            mock_registry_instance.policy = MagicMock()
            
            MockRegistry.from_runtime.return_value = mock_registry_instance
            
            with patch("spectrue_core.verification.pipeline.SearchManager") as MockSearchManagerCls:
                MockSearchManagerCls.return_value = mock_search_mgr
                
                # Mock EmbedService to avoid OpenAI calls
                with patch("spectrue_core.verification.pipeline.EmbedService"):
                    pipeline = ValidationPipeline(config=mock_config, agent=mock_agent)
                    # Manually attach registry if needed, but __init__ does it
                    return pipeline

    @pytest.mark.asyncio
    async def test_deep_mode_triggers_per_claim_scoring(self, pipeline, mock_agent, mock_search_mgr):
        """Verify that search_type='deep' triggers per-claim independent scoring."""
        # Setup: 3 claims
        claims = [
            {"id": "c1", "text": "Claim 1", "search_queries": ["q1"]},
            {"id": "c2", "text": "Claim 2", "search_queries": ["q2"]},
            {"id": "c3", "text": "Claim 3", "search_queries": ["q3"]},
        ]
        mock_agent.extract_claims.return_value = (claims, False, "news", "", "")
        
        # Setup parallel scoring to return dummy result (deep mode uses this)
        mock_agent.score_evidence_parallel.return_value = {
            "verified_score": 0.8,
            "claim_verdicts": [
                {"claim_id": "c1", "verdict": "verified", "verdict_score": 0.9, "rgba": [0.1, 0.9, 0.8, 0.8]},
                {"claim_id": "c2", "verdict": "verified", "verdict_score": 0.8, "rgba": [0.1, 0.8, 0.8, 0.7]},
                {"claim_id": "c3", "verdict": "verified", "verdict_score": 0.7, "rgba": [0.2, 0.7, 0.8, 0.6]},
            ],
            "rationale": "Test rationale",
            "danger_score": 0.1,
            "style_score": 0.8,
            "explainability_score": 0.7,
        }

        # Execute in DEEP mode via pipeline
        await pipeline.execute(
            fact="Test Fact",
            search_type="deep",
            gpt_model="gpt-4",
            lang="en"
        )

        # Deep mode uses score_evidence_parallel, not score_evidence
        assert mock_agent.score_evidence_parallel.call_count == 1
        assert mock_agent.score_evidence.call_count == 0

    @pytest.mark.asyncio
    async def test_smart_mode_uses_batch_scoring(self, pipeline, mock_agent, mock_search_mgr):
        """Verify that search_type='smart' uses standard batch scoring (not parallel)."""
        # Setup: 3 claims
        claims = [
            {"id": "c1", "text": "Claim 1", "search_queries": ["q1"]},
            {"id": "c2", "text": "Claim 2", "search_queries": ["q2"]},
            {"id": "c3", "text": "Claim 3", "search_queries": ["q3"]},
        ]
        mock_agent.extract_claims.return_value = (claims, False, "news", "", "")
        
        # smart mode uses "normal" profile which uses standard score_evidence
        mock_agent.score_evidence.return_value = {
            "verified_score": 0.8, 
            "claim_verdicts": [
                {"claim_id": "c1", "verdict": "verified", "verdict_score": 0.8, "rgba": [0.1, 0.8, 0.8, 0.8]},
            ],
            "rationale": "Batch rationale",
            "danger_score": 0.1,
            "style_score": 0.8,
            "explainability_score": 0.8,
        }

        # Execute in SMART mode (maps to "normal" profile)
        await pipeline.execute(
            fact="Test Fact",
            search_type="smart",
            gpt_model="gpt-4",
            lang="en"
        )

        # Smart mode uses "normal" profile = standard score_evidence
        assert mock_agent.score_evidence.call_count == 1
        assert mock_agent.score_evidence_parallel.call_count == 0
