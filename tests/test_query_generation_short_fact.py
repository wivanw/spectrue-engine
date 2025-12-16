import asyncio

from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.fact_verifier_composite import FactVerifierComposite


def test_short_fact_llm_called_and_fallback_works(monkeypatch):
    """
    M44: LLM query generation is now always called for all facts (including short ones).
    When LLM returns empty, fallback should still produce valid queries.
    """
    # Ensure explicit override is not set.
    monkeypatch.delenv("SPECTRUE_LLM_QUERY_REWRITE_SHORT", raising=False)

    verifier = FactVerifierComposite(
        SpectrueConfig(openai_api_key="test", tavily_api_key="test", google_fact_check_key="test")
    )

    called = {"llm": False, "llm_failed": False}

    async def _mock_extract_claims(*args, **kwargs):
        called["llm"] = True
        # Simulate LLM failure -> should trigger fallback
        called["llm_failed"] = True
        raise Exception("Simulated LLM failure for testing fallback")

    async def _oracle_hit(_query, _lang):
        # Stop early so the test does not need external search providers.
        return {"verified_score": 0.5, "cost": 0, "sources": [], "analysis": "ok", "rgba": [0,0,0,0]}

    monkeypatch.setattr(verifier.agent, "extract_claims", _mock_extract_claims)
    monkeypatch.setattr(verifier.google_tool, "search", _oracle_hit)

    short_fact = "Mars has two moons."
    res = asyncio.run(
        verifier.verify_fact(
            fact=short_fact,
            search_type="basic",
            gpt_model="gpt-5-nano",
            lang="en",
            analysis_mode="general",
            context_text="",
            content_lang="en",
            search_provider="auto",
            max_cost=9999,
        )
    )

    # M44: LLM should be called for all facts
    assert called["llm"] is True, "LLM should be called for query generation"
    # Fallback should have been triggered
    assert called["llm_failed"] is True
    # Result should still be valid (oracle returned successfully)
    assert res["text"] == short_fact

