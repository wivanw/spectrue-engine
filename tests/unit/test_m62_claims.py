"""
M62 Tests: Claim Extraction & Query Generation Refactoring

Tests for:
- Context-aware claim extraction (normalized_text, topic_group, check_worthiness)
- Topic-based round-robin query selection
- Hard caps for scoring
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.agents.skills.claims import ClaimExtractionSkill, TOPIC_GROUPS
from spectrue_core.agents.skills.scoring import ScoringSkill


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
        
        claims, _ = await skill.extract_claims("Some article text", lang="en")
        
        assert len(claims) == 1
        claim = claims[0]
        
        # M62 fields present
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
        
        claims, _ = await skill.extract_claims("Text", lang="en")
        
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
        
        claims, _ = await skill.extract_claims("Text", lang="en")
        
        # Should fallback to importance
        assert claims[0]["check_worthiness"] == 0.7


@pytest.mark.unit
class TestM62QuerySelection:
    """Test typed priority slot query selection."""
    
    @pytest.fixture
    def pipeline(self, mock_config):
        """Create a minimal pipeline by mocking SearchManager."""
        from unittest.mock import patch
        agent = MagicMock()
        
        with patch('spectrue_core.verification.pipeline.SearchManager'):
            pipeline = ValidationPipeline(mock_config, agent)
        return pipeline
    
    def test_select_diverse_queries_typed_slots(self, pipeline):
        """Queries should follow typed priority: Core -> Numeric -> Attribution."""
        claims = [
            {
                "id": "c1",
                "text": "Trump tariffs",
                "topic_group": "Economy",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "search_queries": ["quote query", "Trump announces tariffs China 2025"]
            },
            {
                "id": "c2",
                "text": "GDP grew 5%",
                "topic_group": "Economy",
                "type": "numeric",
                "importance": 0.7,
                "check_worthiness": 0.7,
                "search_queries": ["GDP quote", "US GDP growth 5 percent 2025"]
            },
            {
                "id": "c3",
                "text": "Biden said X",
                "topic_group": "Politics",
                "type": "attribution",
                "importance": 0.6,
                "check_worthiness": 0.6,
                "search_queries": ["Biden quote X", "Biden statement event 2025"]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Should get 3 queries in order: Core first, then Numeric, then Attribution
        assert len(queries) == 3
        # First should be from core claim (Event-based query at index 1)
        assert "Trump" in queries[0] or "tariffs" in queries[0]
    
    def test_select_diverse_queries_skips_sidefacts(self, pipeline):
        """Sidefacts should be completely skipped."""
        claims = [
            {
                "id": "c1",
                "topic_group": "Science",
                "type": "sidefact",  # Should be skipped!
                "importance": 0.4,
                "check_worthiness": 0.5,
                "search_queries": ["sidefact query"]
            },
            {
                "id": "c2",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "search_queries": ["quote", "core event query"]
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
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "search_queries": ["worthy query"]
            },
            {
                "id": "c2",
                "topic_group": "Other",
                "type": "core",
                "importance": 0.2,
                "check_worthiness": 0.2,  # Below threshold
                "search_queries": ["unworthy query"]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Only worthy claim should be used
        assert len(queries) == 1
        assert "worthy" in queries[0]
    
    def test_select_diverse_queries_multi_topic_cores(self, pipeline):
        """Additional Core claims from different topics should fill extra slots."""
        claims = [
            {
                "id": "c1",
                "text": "Black hole discovery",
                "topic_group": "Science",
                "type": "core",
                "importance": 0.9,
                "check_worthiness": 0.9,
                "search_queries": ["quote1", "black hole discovery 2025"]
            },
            {
                "id": "c2",
                "text": "Trump tariffs",
                "topic_group": "Economy",
                "type": "core",  # Different topic!
                "importance": 0.85,
                "check_worthiness": 0.85,
                "search_queries": ["quote2", "Trump tariffs China 2025"]
            },
            {
                "id": "c3",
                "text": "Orbán law",
                "topic_group": "Politics",
                "type": "core",  # Another different topic!
                "importance": 0.8,
                "check_worthiness": 0.8,
                "search_queries": ["quote3", "Orban flag law Hungary 2025"]
            }
        ]
        
        queries = pipeline._select_diverse_queries(claims, max_queries=3)
        
        # Should get 3 queries from 3 different core claims (different topics)
        assert len(queries) == 3
        # All should be Event-based queries (index 1)
        assert "black hole" in queries[0].lower() or "discovery" in queries[0].lower()
        assert "trump" in queries[1].lower() or "tariffs" in queries[1].lower()
        assert "orban" in queries[2].lower() or "flag" in queries[2].lower()


@pytest.mark.unit
class TestM62HardCaps:
    """Test hard caps for scoring."""
    
    @pytest.fixture
    def skill(self, mock_llm_client, mock_config):
        return ScoringSkill(config=mock_config, llm_client=mock_llm_client)
    
    @pytest.mark.asyncio
    async def test_cap_insufficient_domains(self, skill, mock_llm_client):
        """Score should be capped to 0.65 if < 2 independent domains."""
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.9}  # LLM gives high score
            ],
            "rationale": "Good evidence."
        }
        
        pack = {
            "original_fact": "Test",
            "claims": [{"id": "c1", "importance": 1.0, "type": "core"}],
            "search_results": [],
            "metrics": {
                "per_claim": {
                    "c1": {"independent_domains": 1}  # Only 1 domain!
                }
            }
        }
        
        result = await skill.score_evidence(pack)
        
        # Should be capped to 0.65
        assert result["verified_score"] == 0.65
        assert "caps_applied" in result
        assert result["caps_applied"][0]["reason"] == "<2 independent domains (1)"
    
    @pytest.mark.asyncio
    async def test_cap_numeric_no_primary(self, skill, mock_llm_client):
        """Numeric claim without primary source should be capped to 0.60."""
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.85}
            ],
            "rationale": "Figure verified."
        }
        
        pack = {
            "original_fact": "Test with numbers",
            "claims": [{"id": "c1", "importance": 1.0, "type": "numeric"}],
            "search_results": [],
            "metrics": {
                "per_claim": {
                    "c1": {
                        "independent_domains": 5,  # Enough domains
                        "primary_present": False    # But no primary!
                    }
                }
            }
        }
        
        result = await skill.score_evidence(pack)
        
        # Should be capped to 0.60
        assert result["verified_score"] == 0.60
        assert "caps_applied" in result
        assert result["caps_applied"][0]["reason"] == "numeric claim, no primary source"
    
    @pytest.mark.asyncio
    async def test_no_cap_when_sufficient_evidence(self, skill, mock_llm_client):
        """No cap should be applied when evidence is sufficient."""
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.9}
            ],
            "rationale": "Strong evidence."
        }
        
        pack = {
            "original_fact": "Test",
            "claims": [{"id": "c1", "importance": 1.0, "type": "core"}],
            "search_results": [],
            "metrics": {
                "per_claim": {
                    "c1": {
                        "independent_domains": 4,
                        "primary_present": True
                    }
                }
            }
        }
        
        result = await skill.score_evidence(pack)
        
        # No cap, keep original
        assert result["verified_score"] == 0.9
        assert "caps_applied" not in result
