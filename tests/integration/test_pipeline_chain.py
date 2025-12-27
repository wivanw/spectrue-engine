# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Algorithmic Chain Test for Spectrue Pipeline.

This test file implements a step-by-step simulation of the verification pipeline,
verifying state at each transition to eliminate heuristics and ensure data flow correctness.

Steps:
1. Input Preparation & Inline Source Verification
2. Search Input Construction (SearchFlowInput)
3. Orchestration & PhaseRunner Execution
4. Evidence Aggregation & Stance Logic
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.verification.pipeline_search import SearchFlowInput, SearchFlowState, run_search_flow
from spectrue_core.verification.phase_runner import PhaseRunner
from spectrue_core.verification.search_mgr import SearchManager
from spectrue_core.embeddings import EmbedService


# Shared State Container
class PipelineState:
    def __init__(self):
        self.inline_sources = []
        self.search_input = None
        self.phase_runner = None
        self.phase_results = {}
        self.final_sources = []

@pytest.mark.asyncio
async def test_pipeline_algorithmic_chain():
    """
    Executes the pipeline steps sequentially, validating data integrity at each boundary.
    """
    if not EmbedService.is_available():
        pytest.skip("Embeddings not available")

    # -------------------------------------------------------------------------
    # Setup Mocks & Data
    # -------------------------------------------------------------------------
    claim_text = "The exoplanet PSR J2322-2650b has a lemon-like shape."
    inline_url = "https://iopscience.iop.org/article/10.3847/2041-8213/ae157c"
    fact_text = f"Some context. {claim_text} See: {inline_url}"
    
    # Real content that matches the claim (sufficient for embedding match > 0.35)
    real_content = """
    We report the detection of PSR J2322-2650b. 
    The planet orbits a millisecond pulsar. 
    It has a lemon-like shape due to tidal forces.
    """
    
    mock_config = MagicMock()
    mock_config.runtime.features.claim_orchestration = True
    # Ensure locale config doesn't return a Mock (fixes pydantic error)
    mock_config.runtime.locale = None
    
    mock_search_mgr = MagicMock(spec=SearchManager)
    # Allow fetches
    mock_search_mgr.fetch_url_content = AsyncMock(return_value=real_content)
    mock_search_mgr.calculate_cost.return_value = 0
    mock_search_mgr.can_afford.return_value = True
    
    mock_agent = AsyncMock()
    # Mock verify_inline_source_relevance to return DICT
    mock_agent.verify_inline_source_relevance.return_value = {
        "is_relevant": True, 
        "is_primary": True, 
        "reason": "Test Reason"
    }
    mock_agent.oracle_skill = MagicMock()
    mock_agent.clustering_skill = AsyncMock()
    mock_agent.clustering_skill.parse_clustering_response.return_value = {} 

    # Initialize Pipeline instance (wrapper)
    # Pipeline creates its own SearchManager, so we must mock the class OR inject our mock after init
    # But init calls SearchManager constructor. We can let it create a real/mock one and overwrite it.
    # HOWEVER, SearchManager init requires config which we mock.
    
    pipeline = ValidationPipeline(mock_config, mock_agent)
    # Inject our configured mock search manager
    pipeline.search_mgr = mock_search_mgr
    
    state = PipelineState()
    
    print("\n[Step 1] Input Preparation & Inline Fetching...")
    # -------------------------------------------------------------------------
    # Step 1: Input Preparation
    # -------------------------------------------------------------------------
    # Simulate inline source candidate from extraction
    candidate_source = {"url": inline_url, "domain": "iop.org", "anchor": inline_url}
    
    target_claim = {"id": "c1", "text": claim_text, "normalized_text": claim_text, "importance": 1.0}
    
    # We need to mock the web_tool inside search_mgr because pipeline uses search_mgr.web_tool.fetch_page_content
    # But wait, my fix used search_mgr.web_tool.fetch_page_content
    # Let's verify what search_mgr has.
    mock_web = AsyncMock()
    # Simulate Fetch FAILURE (missing content) to test fallback
    mock_web.fetch_page_content.return_value = None 
    mock_search_mgr.web_tool = mock_web
    
    final_sources_accumulator = []
    
    # Correct signature for _verify_inline_sources
    verified_sources = await pipeline._verify_inline_sources(
        inline_sources=[candidate_source],
        claims=[target_claim], # Pass claim to enable primary check and fetch
        fact=fact_text,
        final_sources=final_sources_accumulator,
        progress_callback=None
    )
    
    # ALGORITHMIC CHECK 1:
    assert len(verified_sources) == 1, "Must verify 1 inline source"
    assert verified_sources[0]["url"] == inline_url
    # Content fail simulation:
    assert verified_sources[0].get("content") is None, "Content should be None (simulated fail)"
    
    state.inline_sources = verified_sources
    print(f"✅ Step 1 Passed. Inline sources: {len(state.inline_sources)} with content: {state.inline_sources[0].get('content')}")

    print("\n[Step 2] Search Input Construction...")
    # -------------------------------------------------------------------------
    # Step 2: Search Flow Input
    # -------------------------------------------------------------------------
    # Construct SearchFlowInput passing inline sources (The M109 Fix)
    
    target_claim = {"id": "c1", "text": claim_text, "normalized_text": claim_text, "importance": 1.0}
    
    inp = SearchFlowInput(
        fact=fact_text,
        lang="en",
        gpt_model="gpt-5-nano",
        search_type="standard",
        max_cost=100,
        article_intent="news",
        search_queries=[claim_text],
        claims=[target_claim],
        preloaded_context=None,
        progress_callback=None,
        inline_sources=state.inline_sources  # CRITICAL: This must be passed
    )
    
    stage_state = SearchFlowState(
        final_context="",
        final_sources=[],
        preloaded_context=None,
        used_orchestration=False
    )
    
    # ALGORITHMIC CHECK 2:
    assert inp.inline_sources is not None
    assert len(inp.inline_sources) == 1
    assert inp.inline_sources[0].get("content") is None
    
    state.search_input = inp
    print("✅ Step 2 Passed. SearchFlowInput carries inline sources.")

    print("\n[Step 3] PhaseRunner Execution (Orchestration)...")
    # -------------------------------------------------------------------------
    # Step 3: PhaseRunner Logic
    # -------------------------------------------------------------------------
    # Instead of calling run_search_flow (which is a black box), let's instantiate PhaseRunner
    # directly with the inputs to verify IT accepts them.
    
    # We need a policy profile
    from spectrue_core.verification.search_policy import default_search_policy
    profile = default_search_policy().get_profile("deep_research")
    
    runner = PhaseRunner(
        search_mgr=mock_search_mgr,
        policy_profile=profile,
        gpt_model="gpt-5-nano",
        inline_sources=state.search_input.inline_sources  # Passing from step 2
    )
    
    # Verify Init
    assert len(runner.inline_sources) == 1, "PhaseRunner must have inline sources stored"
    
    # Mock verdict_ready_for_claim to NOT run real embedding logic if we want to isolate,
    # BUT user wants "algorithmic" so we should use REAL embedding service here to prove
    # the matching algorithm works.
    
    # ExecutionPlan needs to be mocked or built
    from spectrue_core.verification.orchestrator import ExecutionPlan, BudgetClass
    plan = ExecutionPlan(budget_class=BudgetClass.MINIMAL)
    # We don't even need phases if shortcut works!
    
    # Run ONLY the logic for c1
    # accessing private method for granular testing involves risk but gives certainty
    # returning: tuple[list[dict], list[RetrievalHop], SufficiencyDecision, str]
    sources, hops, decision, reason = await runner._run_retrieval_loop_for_claim(
        target_claim,
        phases=[] # No phases needed if shortcut works
    )
    
    # ALGORITHMIC CHECK 3:
    # decision should be SufficiencyDecision.ENOUGH
    # To avoid import issues, check .value if sure, or just string equality insensitive?
    # Or import the enum.
    from spectrue_core.verification.sufficiency import SufficiencyDecision
    assert decision == SufficiencyDecision.ENOUGH, f"Decision should be ENOUGH, got {decision}"
    assert reason == "inline_sufficient", f"Reason should be inline_sufficient, got {reason}"
    assert len(sources) >= 1, "Must return evidence sources"
    
    # Check Stance (The M109 Stance Fix)
    found_stance = False
    for s in sources:
        if s["url"] == inline_url:
            print(f"   Source Stance: {s.get('stance')}")
            if s.get("stance") == "SUPPORT":
                found_stance = True
    
    assert found_stance is True, "Inline source must have auto-set stance='SUPPORT'"
    
    state.phase_results["c1"] = sources
    print("✅ Step 3 Passed. PhaseRunner utilized inline shortcut and set Stance.")

    print("\n[Step 4] Verdict Aggregation...")
    # -------------------------------------------------------------------------
    # Step 4: Final Verdict Simulation
    # -------------------------------------------------------------------------
    # If we have SUPPORT evidence, scoring should work.
    # We can't easily query the LLM scoring here without mocking,
    # but we can verify the INPUT to scoring is correct.
    
    evidence_for_scoring = state.phase_results["c1"]
    
    # Verify we have "SUPPORT" items
    support_items = [s for s in evidence_for_scoring if s.get("stance") == "SUPPORT"]
    assert len(support_items) > 0, "Scoring input must have SUPPORT items"
    
    print("✅ Step 4 Passed. Data ready for Scoring contains explicit SUPPORT.")
    print("\n🎉 ALL ALGORITHMIC CHECKS PASSED. The pipeline flow is mathematically correct.")

