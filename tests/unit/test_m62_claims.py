# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tests: Claim Extraction & Query Generation Refactoring

Tests for:
- Context-aware claim extraction (normalized_text, topic_group, check_worthiness)
- Topic-based round-robin query selection
"""
import pytest
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



