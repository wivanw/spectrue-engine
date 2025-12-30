"""
Tests: Claim Extraction & Query Generation Refactoring

Tests for:
- Context-aware claim extraction (normalized_text, topic_group, check_worthiness)
- Topic-based round-robin query selection
"""
import pytest
from unittest.mock import MagicMock
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.agents.skills.claims import ClaimExtractionSkill


@pytest.mark.unit
class TestM62ClaimExtraction:
    """Test context-aware claim extraction."""
    
    @pytest.fixture
    def skill(self, mock_llm_client, mock_config):
        return ClaimExtractionSkill(config=mock_config, llm_client=mock_llm_client)
    
    @pytest.mark.asyncio
    async def test_extract_claims_with_new_fields(self, skill, mock_llm_client):
        """Claims should include normalized_text, topic_group, check_worthiness."""
        mock_llm_client.call_json.return_value = {
            "claims": [{
                "text": "He announced the new tariffs.",
                "normalized_text": "Donald Trump announced new tariffs on China on December 19, 2025.",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.85,
                "topic_group": "Economy",
                "search_queries": ["Trump tariffs China", "Trump China tariffs December 2025"],
                "check_oracle": False
            }]
        }
        
        # extract_claims returns 3-tuple now
        claims, should_check_oracle, article_intent, _ = await skill.extract_claims("Some article text", lang="en")
        
        assert len(claims) == 1
        claim = claims[0]
        
        # fields present
        assert claim["normalized_text"] == "Donald Trump announced new tariffs on China on December 19, 2025."
        assert claim["topic_group"] == "Economy"
        assert claim["check_worthiness"] == 0.85
    
    @pytest.mark.asyncio
    async def test_topic_group_validation(self, skill, mock_llm_client):
        """Invalid topic_group should fallback to 'Other'."""
        mock_llm_client.call_json.return_value = {
            "claims": [{
                "text": "Test",
                "topic_group": "InvalidTopic",  # Not in TOPIC_GROUPS
                "importance": 0.5,
                "search_queries": ["test"]
            }]
        }
        
        # extract_claims returns 3-tuple now
        claims, _, _, _ = await skill.extract_claims("Text", lang="en")
        
        assert claims[0]["topic_group"] == "Other"
    
    @pytest.mark.asyncio
    async def test_check_worthiness_fallback(self, skill, mock_llm_client):
        """If check_worthiness missing, derive from importance."""
        mock_llm_client.call_json.return_value = {
            "claims": [{
                "text": "Test",
                "importance": 0.7,
                # check_worthiness not provided
                "search_queries": ["test"]
            }]
        }
        
        # extract_claims returns 3-tuple now
        claims, _, _, _ = await skill.extract_claims("Text", lang="en")
        
        # Should fallback to importance
        assert claims[0]["check_worthiness"] == 0.7


@pytest.mark.unit
class TestM62QuerySelection:
    """Test Topic-Aware Round-Robin query selection (Coverage Engine)."""
    
    @pytest.fixture
    def pipeline(self, mock_config):
        """Create a minimal pipeline by mocking SearchManager."""
        from unittest.mock import patch
        agent = MagicMock()
        
        with patch('spectrue_core.verification.pipeline.SearchManager'):
            pipeline = ValidationPipeline(mock_config, agent)
        return pipeline
    
    def test_select_diverse_queries_round_robin_topics(self, pipeline):
        """Round-robin should cover all topics before adding depth."""
        claims = [
            {
                "id": "c1",
                "text": "Trump tariffs",
                "topic_key": "Trump Tariffs",
                "topic_group": "Economy",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "Trump announces tariffs China 2025", "role": "CORE", "score": 1.0},
                    {"text": "Trump tariffs 25 percent rate", "role": "NUMERIC", "score": 0.8}
                ]
            },
            {
                "id": "c2",
                "text": "GDP grew 5%",
                "topic_key": "GDP Growth",
                "topic_group": "Economy",
                "type": "numeric",
                "importance": 0.7,
                "check_worthiness": 0.7,
                "query_candidates": [
                    {"text": "US GDP growth 5 percent 2025", "role": "CORE", "score": 1.0}
                ]
            },
            {
                "id": "c3",
                "text": "Biden said X",
                "topic_key": "Biden Statement",
                "topic_group": "Politics",
                "type": "attribution",
                "importance": 0.6,
                "check_worthiness": 0.6,
                "query_candidates": [
                    {"text": "Biden statement event 2025", "role": "CORE", "score": 1.0}
                ]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Should get 3 queries covering 3 different topic_keys
        assert len(queries) == 3
        # First should be from highest importance topic
        assert "trump" in queries[0].lower() or "tariffs" in queries[0].lower()
    
    def test_select_diverse_queries_skips_sidefacts(self, pipeline):
        """Sidefacts should be completely skipped."""
        claims = [
            {
                "id": "c1",
                "topic_key": "Sidefact Topic",
                "topic_group": "Science",
                "type": "sidefact",  # Should be skipped!
                "importance": 0.4,
                "check_worthiness": 0.5,
                "query_candidates": [{"text": "sidefact query", "role": "CORE", "score": 1.0}]
            },
            {
                "id": "c2",
                "topic_key": "Core Topic",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "query_candidates": [{"text": "core event query", "role": "CORE", "score": 1.0}]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Sidefact should be skipped - its query should NOT be present
        all_queries_str = " ".join(queries).lower()
        assert "sidefact" not in all_queries_str
        # Core claim query should be present
        assert "core" in all_queries_str
    
    def test_select_diverse_queries_filters_low_worthiness(self, pipeline):
        """Claims with check_worthiness < 0.4 should be filtered."""
        claims = [
            {
                "id": "c1",
                "topic_key": "Worthy Topic",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "query_candidates": [{"text": "worthy query", "role": "CORE", "score": 1.0}]
            },
            {
                "id": "c2",
                "topic_key": "Unworthy Topic",
                "topic_group": "Other",
                "type": "core",
                "importance": 0.2,
                "check_worthiness": 0.2,  # Below threshold
                "query_candidates": [{"text": "unworthy query", "role": "CORE", "score": 1.0}]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Only worthy claim should be used
        assert len(queries) == 1
        assert "worthy" in queries[0]
    
    def test_select_diverse_queries_multi_topic_cores(self, pipeline):
        """Different topic_keys get round-robin coverage."""
        claims = [
            {
                "id": "c1",
                "text": "Black hole discovery",
                "topic_key": "Black Hole Discovery",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "black hole discovery 2025", "role": "CORE", "score": 1.0}
                ]
            },
            {
                "id": "c2",
                "text": "Trump tariffs",
                "topic_key": "Trump Tariffs",
                "topic_group": "Economy",
                "type": "core",  # Different topic!
                "importance": 0.85,
                "check_worthiness": 0.85,
                "query_candidates": [
                    {"text": "Trump tariffs China 2025", "role": "CORE", "score": 1.0}
                ]
            },
            {
                "id": "c3",
                "text": "Orbán law",
                "topic_key": "Orban Hungary Law",
                "topic_group": "Politics",
                "type": "core",  # Another different topic!
                "importance": 0.8,
                "check_worthiness": 0.8,
                "query_candidates": [
                    {"text": "Orban flag law Hungary 2025", "role": "CORE", "score": 1.0}
                ]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Should get 3 queries from 3 different topic_keys (round-robin coverage)
        assert len(queries) == 3
        # Verify each topic is covered
        all_queries_lower = " ".join(queries).lower()
        assert "black hole" in all_queries_lower or "discovery" in all_queries_lower
        assert "trump" in all_queries_lower or "tariffs" in all_queries_lower
        assert "orban" in all_queries_lower or "hungary" in all_queries_lower
    

    
    def test_fuzzy_dedup_removes_near_identical_queries(self, pipeline):
        """Near-identical queries (90%+ word overlap) should be deduplicated."""
        claims = [
            {
                "id": "c1",
                "topic_key": "NASA Mars",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "query_candidates": [
                    {"text": "NASA Mars mission 2025", "role": "CORE", "score": 1.0},
                    {"text": "Mars mission NASA 2025", "role": "LOCAL", "score": 0.5}  # 100% same words!
                ]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=2)
        
        # Second query should be deduplicated (same words, different order)
        assert len(queries) == 1
    
    def test_legacy_search_queries_fallback(self, pipeline):
        """Claims without query_candidates should fallback to search_queries."""
        claims = [
            {
                "id": "c1",
                "topic_key": "Legacy Topic",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "search_queries": ["Legacy query one", "Legacy query two"]  # Old format!
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=2)
        
        assert len(queries) >= 1
        assert "legacy" in queries[0].lower()

        assert "legacy" in queries[1].lower()
