# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.verification.pipeline import ValidationPipeline

@pytest.mark.unit
class TestOracleOptimization:
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.extract_claims = AsyncMock()
        agent.clean_article = AsyncMock()
        agent.cluster_evidence = AsyncMock()
        agent.score_evidence = AsyncMock()
        agent.verify_oracle_relevance = AsyncMock()
        agent.verify_inline_source_relevance = AsyncMock()
        return agent

    @pytest.fixture
    def mock_search_mgr(self):
        mgr = MagicMock()
        mgr.search_tier1 = AsyncMock(return_value=("", []))
        mgr.search_tier2 = AsyncMock(return_value=("", []))
        mgr.search_unified = AsyncMock(return_value=("", []))
        mgr.check_oracle = AsyncMock(return_value=None)
        mgr.check_oracle_hybrid = AsyncMock(return_value=None)
        mgr.reset_metrics = MagicMock()
        mgr.calculate_cost = MagicMock(return_value=0)
        mgr.can_afford = MagicMock(return_value=True)
        mgr.get_search_meta = MagicMock(return_value={})
        # Initialize counters as ints
        mgr.tavily_calls = 0
        mgr.google_cse_calls = 0
        mgr.page_fetches = 0
        return mgr

    @pytest.fixture
    def pipeline(self, mock_config, mock_agent, mock_search_mgr):
        pipeline = ValidationPipeline(config=mock_config, agent=mock_agent, search_mgr=mock_search_mgr)
        return pipeline

    @pytest.mark.asyncio
    async def test_oracle_optimization_specific_claims(self, pipeline, mock_agent, mock_search_mgr):
        """Test that Oracle checks are performed specifically on flagged claims."""
        
        claims = [
            {"id": "c1", "text": "Claim One is the most important and longest claim here", "check_oracle": True, "importance": 0.9, "check_worthiness": 0.9, "claim_role": "thesis"},
        ]
        # should_check_oracle = True
        mock_agent.extract_claims.return_value = (claims, True, "news", "")
        
        # Oracle returns None (miss)
        mock_search_mgr.check_oracle_hybrid.return_value = None
        
        mock_agent.generate_search_queries = AsyncMock(return_value=["q1", "q2"])
        mock_agent.score_evidence = AsyncMock(return_value={"verified_score": 0.5, "rationale": "test"})
        mock_agent.cluster_evidence = AsyncMock(return_value=[])

        await pipeline.execute(fact="Test fact", lang="en")
        
        # Verify calls to check_oracle_hybrid
        # Limit is now 1 candidate to save quota.
        assert mock_search_mgr.check_oracle_hybrid.call_count == 1
        
        # Check actual queries
        calls = mock_search_mgr.check_oracle_hybrid.call_args_list
        queries = [c[0][0] for c in calls]
        
        # Normalize logic lowercases queries usually
        # Should be Claim One (since it's first eligible)
        assert any("claim one" in q.lower() for q in queries)

    @pytest.mark.asyncio
    async def test_oracle_limit_max_calls(self, pipeline, mock_agent, mock_search_mgr):
        """Test that Oracle checks are strictly limited to 1 call max (Optimization)."""
        
        claims = [
            {"id": "c1", "text": "Claim 1", "check_oracle": True, "importance": 0.9, "check_worthiness": 0.9},
        ]
        # All true
        mock_agent.extract_claims.return_value = (claims, True, "news", "")
        mock_search_mgr.check_oracle_hybrid.return_value = None
        mock_agent.generate_search_queries = AsyncMock(return_value=["q1", "q2"])
        mock_agent.score_evidence = AsyncMock(return_value={"verified_score": 0.5, "rationale": "test"})
        mock_agent.cluster_evidence = AsyncMock(return_value=[])
        
        await pipeline.execute(fact="Limit test", lang="en")
        
        # Limit is 1
        assert mock_search_mgr.check_oracle_hybrid.call_count == 1

    @pytest.mark.asyncio
    async def test_oracle_fallback(self, pipeline, mock_agent, mock_search_mgr):
        """Test fallback to core claim/fact if flag is True but no explicit check_oracle claims."""
        
        claims = [
            {"id": "c1", "text": "Claim 1", "check_oracle": False, "type": "core", "importance": 0.9, "check_worthiness": 0.9},
        ]
        # Flag True (heuristic override)
        mock_agent.extract_claims.return_value = (claims, True, "news", "")
        mock_search_mgr.check_oracle_hybrid.return_value = None
        mock_agent.generate_search_queries = AsyncMock(return_value=["q1", "q2"])
        mock_agent.score_evidence = AsyncMock(return_value={"verified_score": 0.5, "rationale": "test"})
        mock_agent.cluster_evidence = AsyncMock(return_value=[])
        
        await pipeline.execute(fact="Fallback test", lang="en")
        
        # Should fallback to core claim 'Claim 1'
        assert mock_search_mgr.check_oracle_hybrid.call_count == 1
        call_args = mock_search_mgr.check_oracle_hybrid.call_args[0][0]
        assert "claim 1" in call_args.lower()
