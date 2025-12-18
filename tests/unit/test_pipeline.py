
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline

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
        agent.verify_inline_source_relevance = AsyncMock()  # T7
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
    async def test_smart_mode_waterfall(self, pipeline, mock_agent, mock_search_mgr):
        """T9: Verify Smart Mode uses Waterfall (no parallel T2) even if 'advanced' is requested."""
        mock_agent.extract_claims.return_value = ([], False)
        mock_agent.score_evidence.return_value = {}
        
        # Setup T1 to return GOOD results (> 2 sources) so T2 fallback is NOT triggered
        mock_search_mgr.search_tier1.return_value = (
            "Context T1", 
            [{"url": "http://t1a.com"}, {"url": "http://t1b.com"}]
        )
        
        # search_type="advanced" used to trigger parallel T2. Now it should be ignored (Smart Mode).
        await pipeline.execute(fact="X", search_type="advanced", gpt_model="gpt-4", lang="en")
        
        assert mock_search_mgr.search_tier1.called
        # T2 should NOT be called because T1 yielded results and parallel is disabled
        assert not mock_search_mgr.search_tier2.called

    @pytest.mark.asyncio
    async def test_inline_source_verification_t7(self, pipeline, mock_agent, mock_search_mgr):
        """T7: Test inline source relevance verification against claims."""
        # Input text with inline URL (will be extracted as inline source)
        # URL in parentheses format matches Pattern 2 in _extract_url_anchors
        text_with_url = "Trump announced blockade (https://example.com/statement/12345) of Venezuela oil"
        
        # Mock claims extraction with a relevant claim
        mock_agent.extract_claims.return_value = (
            [{"id": "c1", "text": "Trump announced blockade of Venezuela", "type": "core", "search_queries": ["trump venezuela blockade"]}],
            False
        )
        
        # Mock inline source verification - return is_primary=True
        mock_agent.verify_inline_source_relevance.return_value = {
            "is_relevant": True,
            "is_primary": True,  # Official statement source is primary
            "reason": "Official statement from the person quoted in the claim"
        }
        
        mock_agent.score_evidence.return_value = {"verified_score": 0.9}
        
        result = await pipeline.execute(
            fact=text_with_url,
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Verify inline source verification was called (at least once for the URL found)
        assert mock_agent.verify_inline_source_relevance.called
        
        # Check that inline source was added to sources with is_primary=True
        inline_sources = [s for s in result["sources"] if s.get("source_type") == "inline"]
        assert len(inline_sources) >= 1
        assert inline_sources[0]["is_primary"] is True
        assert inline_sources[0]["is_trusted"] is True  # Primary sources are trusted

    @pytest.mark.asyncio
    async def test_inline_source_rejected_when_not_relevant(self, pipeline, mock_agent, mock_search_mgr):
        """T7: Test that irrelevant inline sources are excluded."""
        text_with_url = "News article with author link (https://twitter.com/author123)"
        
        mock_agent.extract_claims.return_value = (
            [{"id": "c1", "text": "Climate change accelerates", "type": "core"}],
            False
        )
        
        # Mock inline source verification - author's Twitter is NOT relevant to climate claim
        mock_agent.verify_inline_source_relevance.return_value = {
            "is_relevant": False,
            "is_primary": False,
            "reason": "This is the article author's social media, not related to the claim"
        }
        
        mock_agent.score_evidence.return_value = {"verified_score": 0.5}
        
        result = await pipeline.execute(
            fact=text_with_url,
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Verify inline source was rejected (not in sources)
        inline_sources = [s for s in result["sources"] if s.get("source_type") == "inline"]
        assert len(inline_sources) == 0
