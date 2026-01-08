# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

"""Integration tests for Claim Extraction Fallback."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from spectrue_core.agents.skills.claims import ClaimExtractionSkill
from spectrue_core.llm.failures import LLMFailureKind


@pytest.mark.asyncio
async def test_claim_extraction_uses_fallback_on_failure():
    """
    Integration test verifying that ClaimExtractionSkill switches to fallback
    when primary LLM call fails with a fallback-eligible error.
    """
    # Setup dependencies
    mock_config = MagicMock()
    # Mock runtime config for fallback model
    mock_config.runtime.llm.model_claim_extraction = "deepseek-chat"
    mock_config.runtime.llm.model_claim_extraction_fallback = "gpt-5.2"
    
    mock_client = AsyncMock()
    
    # Instantiate skill
    skill = ClaimExtractionSkill(mock_config, mock_client)
    
    # Mock text and chunking
    text = "Sky is blue. Grass is green."
    
    # Mock chunk_for_claims to return a single chunk
    with patch.object(skill, "_chunk_for_claims") as mock_chunk:
        from spectrue_core.utils.text_chunking import TextChunk
        mock_chunk.return_value = ([TextChunk("ck1", text, 0, len(text))], text)
        
        # Mock LLM client call_json
        # First call (DeepSeek) fails with Connection Error
        # Second call (GPT-5.2) succeeds
        
        # Note: call_json is called for EACH chunk. Here we have 1 chunk.
        # But wait, logic is: call_with_fallback calls primary, then fallback.
        # So call_json will be called twice.
        
        # We need to distinguish calls by model arg
        async def side_effect(*args, **kwargs):
            model = kwargs.get("model")
            if model == "deepseek-chat":
                raise Exception("Connection reset by peer")
            elif model == "gpt-5.2":
                return {
                    "claims": [
                        {
                            "claim_text": "Sky is blue",
                            "subject_entities": ["Sky"],
                            "retrieval_seed_terms": ["sky", "blue", "atmosphere"],
                            "falsifiability": {"is_falsifiable": True},
                            "time_anchor": {"type": "permanent"},
                            "predicate_type": "existence",
                        }
                    ],
                    "article_intent": "educational"
                }
            return {}

        mock_client.call_json.side_effect = side_effect
        
        # Also need to mock _enrich_claim to avoid more calls
        with patch.object(skill, "_enrich_claim", return_value=None):
            
            # Execute
            claims, _, _, _ = await skill.extract_claims(text)
            
            # Verify results
            assert len(claims) == 0  # 0 because _enrich_claim returns None
            # But we care about the CALLS to LLM
            
            # Verify LLM calls
            # Expect at least 2 calls: 1 primary (failed), 1 fallback (ok)
            assert mock_client.call_json.call_count >= 2
            
            calls = mock_client.call_json.call_args_list
            models_called = [c.kwargs.get("model") for c in calls]
            
            assert "deepseek-chat" in models_called
            assert "gpt-5.2" in models_called
