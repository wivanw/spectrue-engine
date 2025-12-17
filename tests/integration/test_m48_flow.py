import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.verification.fact_verifier_composite import FactVerifierComposite
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.llm_client import LLMClient

@pytest.fixture
def mock_config():
    # Don't use spec=SpectrueConfig to avoid strict attribute checking on nested mocks during setup
    config = MagicMock()
    # IMPORTANT: Use strings to avoid 400 Bad Request from real tools if they are instantiated
    config.openai_api_key = "test_key" 
    config.tavily_api_key = "test_key"
    config.google_fact_check_key = "test_key"
    config.google_search_api_key = "test_key" # Added string key
    config.google_search_cse_id = "test_id"
    config.openai_model = "gpt-5.2"
    
    # Mock runtime config structure
    config.runtime.llm.timeout_sec = 60.0
    config.runtime.llm.nano_timeout_sec = 20.0 # Added missing field
    config.runtime.llm.cluster_timeout_sec = 30.0
    config.runtime.llm.max_output_tokens_general = 1000
    config.runtime.llm.nano_max_output_tokens = 500
    config.runtime.search.google_cse_cost = 0
    config.runtime.tunables.langdetect_min_prob = 0.8
    config.runtime.tunables.max_claims_deep = 2
    
    # Ensure nested objects are also Mocks
    config.runtime.features.trace_enabled = False
    return config

@pytest.mark.asyncio
async def test_verification_uses_score_evidence(mock_config, caplog):
    """
    Verifies that verify_fact uses the score_evidence pipeline
    with claim extraction, clustering, and LLM-based scoring.
    """
    # Enable visibility of logs
    import logging
    caplog.set_level(logging.DEBUG)

    verifier = FactVerifierComposite(mock_config)
    
    # Mock search tools to return immediate results
    # Return 3+ sources to be considered "good" and avoid Google CSE fallback
    # M48: Use "is_trusted": True to ensure confidence cap allows > 0.5 score
    # Use unique domains to satisfy "independent sources" requirement (typically 2+)
    mock_sources = [{"url": f"http://example{i}.com/page", "title": f"Source {i}", "relevance_score": 0.9, "is_trusted": True} for i in range(5)]
    # Mock search tools
    # M50: Support new Pipeline structure or fall back to legacy
    if hasattr(verifier, "pipeline") and verifier.pipeline:
        target_web = verifier.pipeline.search_mgr.web_tool
        target_oracle = verifier.pipeline.search_mgr.oracle_tool
    else:
        target_web = verifier.search_tool
        target_oracle = verifier.google_tool

    target_web.search = AsyncMock(return_value=("Context", mock_sources))
    target_oracle.search = AsyncMock(return_value=None) # Oracle miss
    
    # Mock agent's llm_client
    verifier.agent.llm_client.call_json = AsyncMock()
    
    # Setup varying responses for different calls
    
    # 1. Claim Extraction Response
    claim_resp = {
        "claims": [{
            "text": "The sky is blue.",
            "type": "core",
            "search_queries": ["sky blue"]
        }]
    }
    
    # 2. Clustering Response (T168)
    cluster_resp = {
        "mappings": [
            {
                "result_index": 1,
                "claim_id": "c1",
                "stance": "support",
                "relevance": 0.9
            },
            {
                "result_index": 2,
                "claim_id": "c1",
                "stance": "support",
                "relevance": 0.9
            },
            {
                "result_index": 3,
                "claim_id": "c1",
                "stance": "support",
                "relevance": 0.9
            }
        ]
    }
    
    # 3. Score Evidence Response (M48 / T164)
    score_resp = {
        "verified_score": 0.9,
        "danger_score": 0.0,
        "rationale": "M48 logic working.",
        "explainability_score": 0.8,
        "style_score": 0.9
    }

    # Side effect to return different responses based on trace_kind or order
    async def side_effect(*args, **kwargs):
        kind = kwargs.get("trace_kind")
        if kind == "claim_extraction":
            return claim_resp
        elif kind == "stance_clustering":
            return cluster_resp
        elif kind == "score_evidence":
            return score_resp
        return {}

    verifier.agent.llm_client.call_json.side_effect = side_effect
    
    # Execute verification
    result = await verifier.verify_fact("The sky is blue", "advanced", "gpt-5.2", "en")
    
    # Assert result
    # M50: Cap logic removed - LLM now has full discretion. Score should match mock response (0.9).
    assert result["verified_score"] == 0.9, f"Expected 0.9, got {result.get('verified_score')}. Logs:\n" + "\n".join(caplog.messages)
    assert result["rationale"] == "M48 logic working."
    assert "cost" in result
    
    # Verify LLM calls
    # Should call extract_claims, cluster_evidence, and score_evidence
    calls = verifier.agent.llm_client.call_json.call_args_list
    kinds = [c.kwargs.get("trace_kind") for c in calls]
    
    assert "claim_extraction" in kinds
    # clustering might be skipped if only 1 claim and few sources? 
    # Logic in _get_final_analysis: if claims and sources_list: await cluster_evidence
    # We returned 5 sources.
    assert "stance_clustering" in kinds, f"Kinds found: {kinds}. Logs:\n" + "\n".join(caplog.messages)
    assert "score_evidence" in kinds, f"Kinds found: {kinds}. Logs:\n" + "\n".join(caplog.messages)
