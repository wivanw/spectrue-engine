
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.verification.pipeline_evidence import (
    _explainability_factor_for_tier,
    _TIER_A_BASELINE,
)


def _apply_explainability_tier_cap(
    score: float,
    tier_counts: dict[str, int],
) -> tuple[float, dict]:
    """Test helper: Apply tier-based explainability cap to a score.

    This is a test-only helper that wraps the production logic
    for easier unit testing of tier cap behavior.
    """
    # Find best tier from counts (A > A' > B > C > D)
    tier_priority = ["A", "A'", "B", "C", "D"]
    best_tier: str | None = None
    for tier in tier_priority:
        if tier_counts.get(tier, 0) > 0:
            best_tier = tier
            break

    factor, source, prior = _explainability_factor_for_tier(best_tier)
    post_score = max(0.0, min(1.0, score * factor))

    trace = {
        "best_tier": best_tier,
        "pre_A": score,
        "prior": prior,
        "baseline": _TIER_A_BASELINE,
        "factor": factor,
        "post_A": post_score,
        "source": source,
    }
    return post_score, trace


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
        agent.verify_search_relevance = AsyncMock(return_value={"is_relevant": True, "reason": "Test ok"}) # M66
        return agent

    @pytest.fixture
    def mock_search_mgr(self):
        mgr = MagicMock()
        mgr.search_tier1 = AsyncMock(return_value=("Context T1", [{"url": "http://t1.com", "content": "c1"}]))
        mgr.search_tier2 = AsyncMock(return_value=("Context T2", [{"url": "http://t2.com", "content": "c2"}]))
        # Add search_unified mock
        mgr.search_unified = AsyncMock(return_value=("Unified Context", [{"url": "http://unified.com", "content": "u1"}]))
        mgr.search_google_cse = AsyncMock(return_value=("CSE Context", [{"link": "http://cse.com", "snippet": "cse1", "title": "CSE"}]))
        mgr.check_oracle = AsyncMock(return_value=None)
        mgr.check_oracle_hybrid = AsyncMock(return_value={"status": "MISS", "relevance_score": 0.1})
        mgr.fetch_url_content = AsyncMock(return_value="Fetched content")
        # Add async mock for evidence acquisition ladder
        mgr.apply_evidence_acquisition_ladder = AsyncMock(return_value=[])
        mgr.calculate_cost = MagicMock(return_value=10)
        mgr.can_afford = MagicMock(return_value=True)
        mgr.get_search_meta = MagicMock(return_value={"total": 1})
        mgr.reset_metrics = MagicMock()
        mgr.tavily_calls = 0  # mock attribute
        return mgr

    @pytest.fixture
    def pipeline(self, mock_config, mock_agent, mock_search_mgr):
        with patch("spectrue_core.verification.pipeline.SearchManager") as MockSearchManagerCls:
            MockSearchManagerCls.return_value = mock_search_mgr
            pipeline = ValidationPipeline(config=mock_config, agent=mock_agent)
            return pipeline

    @pytest.mark.asyncio
    async def test_execute_basic_flow(self, pipeline, mock_agent, mock_search_mgr):
        # Setup Agent mocks
        mock_agent.extract_claims.return_value = (
            [{"id": "c1", "text": "Claim", "search_queries": ["q1"]}],
            False,
            "news",
            "",
        )
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
        # Should call search_unified instead of search_tier1
        mock_search_mgr.search_unified.assert_called()
        # Should not check oracle as should_check_oracle=False and intent="news" isn't strictly forcing it in this mock unless intent logic
        # But wait, news intent triggers Oracle Check in logic.
        # Let's adjust mock intent or assert oracle checked.
        # "news" -> ORACLE_CHECK_INTENT is True. So it SHOULD check oracle.
        # Original test asserted not checked.
        # Let's verify M63/behavior: Intent="news" -> Check Oracle.
        # So check_oracle_hybrid should be called.
        mock_search_mgr.check_oracle_hybrid.assert_called()
        
        # Without quoted evidence in sources, Bayesian scoring defaults to 0.5 (uncertain)
        # The LLM's verified_score is now only one signal, not the final score
        assert result["verified_score"] == 0.5
        # Sources may be empty if evidence flow doesn't process them (mocked pipeline)
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_execute_with_oracle_hit(self, pipeline, mock_agent, mock_search_mgr):
        # Setup Agent mocks: claims detection triggers oracle check
        mock_agent.extract_claims.return_value = ([{"id": "c1", "text": "Fake news"}], True, "news", "") 
        
        # Setup Oracle hit (Jackpot)
        mock_search_mgr.check_oracle_hybrid.return_value = {
            "status": "REFUTED",
            "is_jackpot": True,
            "relevance_score": 0.95, 
            "url": "http://snopes.com/fact",
            "title": "Debunked",
            "summary": "Debunked by Snopes"
        }
        
        # Update unified search mock to return strong evidence for the second call
        mock_search_mgr.search_unified.return_value = (
            "Unified Context", 
            [{
                "url": "http://unified.com", 
                "content": "u1", 
                "stance": "SUPPORT", 
                "quote": "Jackpot evidence found.", 
                "is_trusted": True
            }]
        )

        result = await pipeline.execute(
            fact="Fake news",
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Should stop early
        mock_search_mgr.check_oracle_hybrid.assert_called()
        mock_search_mgr.search_unified.assert_not_called() # Should skipped search
        assert result["verified_score"] == 0.1 # REFUTED -> 0.1

    @pytest.mark.asyncio
    async def test_waterfall_fallback(self, pipeline, mock_agent, mock_search_mgr):
        # Unified returns nothing
        mock_search_mgr.search_unified.return_value = ("", [])
        mock_agent.extract_claims.return_value = ([{"search_queries": ["q1"]}], False, "news", "")
        mock_agent.score_evidence.return_value = {}
        # Enable usage of CSE by ensuring tavily calls > 0 (as logic checks usage before fallback)
        mock_search_mgr.tavily_calls = 1 
        
        await pipeline.execute(fact="X", search_type="standard", gpt_model="gpt-4", lang="en")
        
        # Should call Unified first
        mock_search_mgr.search_unified.assert_called()
        # Should call CSE because Unified was empty
        mock_search_mgr.search_google_cse.assert_called()

    @pytest.mark.asyncio
    async def test_smart_mode_waterfall(self, pipeline, mock_agent, mock_search_mgr):
        """T9/Verify Smart Mode uses Unified Search."""
        mock_agent.extract_claims.return_value = ([{"search_queries": ["q1"]}], False, "news", "")
        mock_agent.score_evidence.return_value = {}
        
        # Setup Unified to return GOOD results
        mock_search_mgr.search_unified.return_value = (
            "Unified Context", 
            [{"url": "http://u1.com"}]
        )
        
        await pipeline.execute(fact="X", search_type="advanced", gpt_model="gpt-4", lang="en")
        
        assert mock_search_mgr.search_unified.called
        # CSE fallback shouldn't be called
        assert not mock_search_mgr.search_google_cse.called

    @pytest.mark.asyncio
    async def test_inline_source_verification_t7(self, pipeline, mock_agent, mock_search_mgr):
        """T7: Test inline source relevance verification against claims."""
        # Input text with inline URL (will be extracted as inline source)
        # URL in parentheses format matches Pattern 2 in _extract_url_anchors
        text_with_url = "Trump announced blockade (https://example.com/statement/12345) of Venezuela oil"
        
        # Mock claims extraction with a relevant claim
        mock_agent.extract_claims.return_value = (
            [{"id": "c1", "text": "Trump announced blockade of Venezuela", "type": "core", "search_queries": ["trump venezuela blockade"]}],
            False,
            "news",
            "",
        )
        
        # Mock inline source verification - return is_primary=True
        mock_agent.verify_inline_source_relevance.return_value = {
            "is_relevant": True,
            "is_primary": True,  # Official statement source is primary
            "reason": "Official statement from the person quoted in the claim"
        }
        
        # Mock apply_evidence_acquisition_ladder to return inline sources
        mock_inline_source = {
            "url": "https://example.com/statement/12345",
            "source_type": "inline",
            "is_primary": True,
            "is_trusted": True,
            "content": "Official statement content",
        }
        mock_search_mgr.apply_evidence_acquisition_ladder = AsyncMock(return_value=[mock_inline_source])
        
        mock_agent.score_evidence.return_value = {"verified_score": 0.9}
        
        await pipeline.execute(
            fact=text_with_url,
            search_type="standard",
            gpt_model="gpt-4",
            lang="en"
        )
        
        # Verify inline source verification was called (at least once for the URL found)
        assert mock_agent.verify_inline_source_relevance.called
        
        # Check apply_evidence_acquisition_ladder was called for inline sources
        assert mock_search_mgr.apply_evidence_acquisition_ladder.called

    @pytest.mark.asyncio
    async def test_inline_source_rejected_when_not_relevant(self, pipeline, mock_agent, mock_search_mgr):
        """T7: Test that irrelevant inline sources are excluded."""
        text_with_url = "News article with author link (https://twitter.com/author123)"
        
        mock_agent.extract_claims.return_value = (
            [{"id": "c1", "text": "Climate change accelerates", "type": "core"}],
            False,
            "news",
            "",
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

    @pytest.mark.asyncio
    async def test_semantic_gating_rejection(self, pipeline, mock_agent, mock_search_mgr):
        """Verify pipeline stops if search results are semantically irrelevant."""
        mock_agent.extract_claims.return_value = ([{"id": "c1", "search_queries": ["q1"], "search_method": "news"}], False, "news", "")
        
        # Search returns results
        mock_search_mgr.search_unified.return_value = ("Context", [{"title": "Irrelevant", "snippet": "Foo"}])
        
        # Gating REJECTS them
        mock_agent.verify_search_relevance.return_value = {
            "is_relevant": False,
            "reason": "Off-topic"
        }
        
        result = await pipeline.execute(fact="Test", search_type="standard", gpt_model="nano", lang="en")
        
        # Verify result is failure
        assert result["verified_score"] == 0.0
        assert "irrelevant" in result["analysis"].lower()
        assert len(result["sources"]) == 0
        
        # Verify heavy steps skipped
        mock_agent.score_evidence.assert_not_called()


def test_explainability_tier_cap_applies() -> None:
    score = 0.9
    post_a, trace_a = _apply_explainability_tier_cap(score, {"A": 1})
    post_c, trace_c = _apply_explainability_tier_cap(score, {"C": 1})

    assert trace_a["best_tier"] == "A"
    assert trace_c["best_tier"] == "C"
    assert post_a > post_c
