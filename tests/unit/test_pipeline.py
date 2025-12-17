
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.verification.evidence_pack import Claim

@pytest.mark.unit
class TestValidationPipeline:
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.extract_claims = AsyncMock()
        agent.clean_article = AsyncMock()
        agent.cluster_evidence = AsyncMock()
        agent.score_evidence = AsyncMock()
        agent.verify_oracle_relevance = AsyncMock()
        return agent

    @pytest.fixture
    def mock_search_mgr(self):
        mgr = MagicMock()
        mgr.search_tier1 = AsyncMock(return_value=("Context T1", [{"url": "http://t1.com", "content": "c1"}]))
        mgr.search_tier2 = AsyncMock(return_value=("Context T2", [{"url": "http://t2.com", "content": "c2"}]))
        mgr.check_oracle = AsyncMock(return_value=None)
        mgr.fetch_url_content = AsyncMock(return_value="Fetched content")
        mgr.calculate_cost = MagicMock(return_value=10)
        mgr.can_afford = MagicMock(return_value=True)
        mgr.get_search_meta = MagicMock(return_value={"total": 1})
        mgr.reset_metrics = MagicMock()
        return mgr

    @pytest.fixture
    def pipeline(self, mock_config, mock_agent, mock_search_mgr):
        # We need to patch SearchManager class or inject the instance
        # ValidationPipeline instantiates SearchManager internally in __init__
        # So we must patch the class where it is imported in pipeline.py
        
        with patch("spectrue_core.verification.pipeline.SearchManager") as MockSearchManagerCls:
            MockSearchManagerCls.return_value = mock_search_mgr
            
            pipeline = ValidationPipeline(config=mock_config, agent=mock_agent)
            return pipeline

    @pytest.mark.asyncio
    async def test_execute_basic_flow(self, pipeline, mock_agent, mock_search_mgr):
        # Setup Agent mocks
        mock_agent.extract_claims.return_value = ([{"id": "c1", "text": "Claim", "search_queries": ["q1"]}], False)
        mock_agent.score_evidence.return_value = {"verified_score": 0.8, "rationale": "ok"}
        
        # Run
        result = await pipeline.execute(
            fact="Sky is blue",
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Verify
        mock_agent.extract_claims.assert_called_once()
        mock_search_mgr.search_tier1.assert_called()
        # Should not check oracle as should_check_oracle=False
        mock_search_mgr.check_oracle.assert_not_called()
        
        assert result["verified_score"] == 0.8
        assert len(result["sources"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_with_oracle_hit(self, pipeline, mock_agent, mock_search_mgr):
        # Setup Agent mocks: claims detection triggers oracle check
        mock_agent.extract_claims.return_value = ([{"id": "c1"}], True) # should_check_oracle=True
        
        # Setup Oracle hit
        mock_search_mgr.check_oracle.return_value = {
            "verified_score": 0.1, 
            "rationale": "Debunked by Snopes"
        }
        # Setup Relevance check
        mock_agent.verify_oracle_relevance.return_value = True
        
        result = await pipeline.execute(
            fact="Fake news",
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Should stop early
        mock_search_mgr.check_oracle.assert_called()
        mock_search_mgr.search_tier1.assert_not_called() # Should skipped search
        assert result["verified_score"] == 0.1
        assert result["rationale"] == "Debunked by Snopes"

    @pytest.mark.asyncio
    async def test_execute_url_resolution(self, pipeline, mock_agent, mock_search_mgr):
        # Input is URL
        url = "http://fake.com/news"
        # Must be > 50 chars to avoid filtering
        long_text = "Resolved Article Text " * 5
        mock_search_mgr.fetch_url_content.return_value = long_text
        mock_agent.clean_article.return_value = "Cleaned Text"
        mock_agent.extract_claims.return_value = ([], False)
        mock_agent.score_evidence.return_value = {"verified_score": 0.5}

        await pipeline.execute(fact=url, search_type="standard", gpt_model="gpt-4", lang="en")
        
        mock_search_mgr.fetch_url_content.assert_called_with(url)
        mock_agent.clean_article.assert_called_with(long_text)
        # Extract claims should differ called with Cleaned Text
        args, _ = mock_agent.extract_claims.call_args
        assert "Cleaned Text" in args[0]

    @pytest.mark.asyncio
    async def test_waterfall_fallback(self, pipeline, mock_agent, mock_search_mgr):
        # T1 returns nothing
        mock_search_mgr.search_tier1.return_value = ("", [])
        mock_agent.extract_claims.return_value = ([], False)
        mock_agent.score_evidence.return_value = {}
        
        await pipeline.execute(fact="X", search_type="standard", gpt_model="gpt-4", lang="en")
        
        # Should call T2 because T1 was empty
        mock_search_mgr.search_tier2.assert_called()

    @pytest.mark.asyncio
    async def test_parallel_execution(self, pipeline, mock_agent, mock_search_mgr):
        mock_agent.extract_claims.return_value = ([], False)
        mock_agent.score_evidence.return_value = {}
        
        # search_type="advanced" -> triggers parallel T2
        await pipeline.execute(fact="X", search_type="advanced", gpt_model="gpt-4", lang="en")
        
        assert mock_search_mgr.search_tier1.called
        assert mock_search_mgr.search_tier2.called
