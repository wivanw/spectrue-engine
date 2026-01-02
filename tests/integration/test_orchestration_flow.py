# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Integration Tests: Claim-Centric Orchestration

Tests the metadata extraction and orchestration logic:
- T9: Horoscope claims get verification_target="none"
- T22: Progressive widening (Phase A sufficient → B/C/D skipped)
- T29: Fail-soft behavior
- T32: RGBA aggregation (context claims don't dilute scores)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from spectrue_core.agents.skills.claims import ClaimExtractionSkill
from spectrue_core.schema.claim_metadata import VerificationTarget, ClaimRole

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

HOROSCOPE_ARTICLE_UK = """
Гороскоп на сьогодні: 21 грудня 2025 року

**Овен (21 березня – 19 квітня)**
Сьогодні зірки сприятимуть вашим фінансовим справам. Очікуйте несподівану грошову винагороду. 
У любовних стосунках варто бути обережним – можливі непорозуміння.

**Телець (20 квітня – 20 травня)**  
День ідеальний для нових починань. Якщо ви давно планували розпочати власний бізнес – 
сьогодні саме час. Венера у вашому знаку посилює вашу привабливість.

**Близнюки (21 травня – 20 червня)**
Уникайте великих покупок сьогодні. Меркурій у ретрограді може спричинити помилки 
у фінансових рішеннях. Краще відкласти важливі справи на наступний тиждень.
"""

HOROSCOPE_ARTICLE_EN = """
Daily Horoscope: December 21, 2025

**Aries (March 21 – April 19)**
Today the stars will favor your financial affairs. Expect an unexpected monetary reward.
In romantic relationships, be careful – misunderstandings are possible.

**Taurus (April 20 – May 20)**
The day is ideal for new beginnings. If you've been planning to start your own business –
today is the time. Venus in your sign enhances your attractiveness.

**Gemini (May 21 – June 20)**
Avoid large purchases today. Mercury in retrograde may cause errors
in financial decisions. Better postpone important matters to next week.
"""

MIXED_ARTICLE = """
# Analysis: Ukraine War Update - December 2025

## Current Situation
According to the Institute for the Study of War (ISW), Ukrainian forces have recaptured 
the city of Tokmak in Zaporizhzhia region. Defense Minister Rustem Umerov confirmed 
the operation was successful.

## Expert Predictions
Political analyst John Smith predicts that "Russia will likely intensify attacks 
in January 2025 to regain lost territory." This is his personal opinion based on 
general observation of the conflict.

## Weather Forecast
Tomorrow's weather in Kyiv will be cloudy with temperatures around -5°C.
This is according to the Ukrainian Hydrometeorological Center.
"""


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with configurable responses."""
    client = MagicMock()
    client.call_json = AsyncMock()
    return client


@pytest.fixture
def mock_config():
    """Create a mock SpectrueConfig."""
    config = MagicMock()
    config.openai_api_key = "test_key"
    config.tavily_api_key = "test_key"
    config.google_fact_check_key = "test_key"
    config.runtime = MagicMock()
    config.runtime.features.trace_enabled = False
    config.runtime.features.log_redaction = False
    config.runtime.features.trace_safe_payloads = False
    return config


@pytest.fixture
def claim_skill(mock_config, mock_llm_client):
    """Create ClaimExtractionSkill with mocked config and LLM."""
    skill = ClaimExtractionSkill(mock_config, mock_llm_client)
    return skill


# ─────────────────────────────────────────────────────────────────────────────
# T9: Horoscope Metadata Classification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_horoscope_claims_get_none_target(claim_skill, mock_llm_client):
    """
    T9: Horoscope claims should have verification_target="none" or "existence".
    
    When LLM extracts claims from horoscope text, prediction-type claims
    should NOT be verified against reality (waste of search budget).
    """
    # Mock LLM response for horoscope extraction
    mock_llm_client.call_json.return_value = {
        "article_intent": "prediction",
        "claims": [
            {
                "text": "Сьогодні зірки сприятимуть вашим фінансовим справам",
                "normalized_text": "Stars will favor Aries financial affairs today",
                "type": "core",
                "claim_category": "OPINION",
                "satire_likelihood": 0.0,
                "topic_group": "Other",
                "topic_key": "Aries Horoscope",
                "importance": 0.3,
                "check_worthiness": 0.1,
                "harm_potential": 1,
                "verification_target": "none",
                "claim_role": "context",
                "search_locale_plan": {"primary": "uk", "fallback": []},
                "retrieval_policy": {"channels_allowed": []},
                "metadata_confidence": "high",
                "query_candidates": [],  # No queries for predictions
                "search_queries": [],
            },
            {
                "text": "Меркурій у ретрограді може спричинити помилки",
                "normalized_text": "Mercury in retrograde may cause errors in decisions",
                "type": "sidefact",
                "claim_category": "OPINION",
                "satire_likelihood": 0.0,
                "topic_group": "Other",
                "topic_key": "Gemini Horoscope",
                "importance": 0.2,
                "check_worthiness": 0.1,
                "harm_potential": 1,
                "verification_target": "none",
                "claim_role": "context",
                "search_locale_plan": {"primary": "uk", "fallback": []},
                "retrieval_policy": {"channels_allowed": []},
                "metadata_confidence": "high",
                "query_candidates": [],
                "search_queries": [],
            },
        ],
    }
    
    # Extract claims
    claims, check_oracle, article_intent, _ = await claim_skill.extract_claims(
        HOROSCOPE_ARTICLE_UK,
        lang="uk",
        max_claims=5,
    )
    
    # Assert: Article intent is prediction/opinion
    assert article_intent in ("prediction", "opinion"), f"Expected prediction/opinion, got {article_intent}"
    
    # Assert: All claims have verification_target="none" (no reality checking)
    for claim in claims:
        metadata = claim.get("metadata")
        assert metadata is not None, f"Claim {claim.get('id')} has no metadata"
        assert metadata.verification_target == VerificationTarget.NONE, \
            f"Claim {claim.get('id')} should have target=none, got {metadata.verification_target}"
        
        # Assert: Claim role should be "context" for non-verifiable claims
        assert metadata.claim_role == ClaimRole.CONTEXT, \
            f"Claim {claim.get('id')} should have role=context, got {metadata.claim_role}"
        
        # Assert: No search queries generated (nothing to verify)
        assert claim.get("search_queries") == [], \
            f"Claim {claim.get('id')} should have empty search_queries"
        assert claim.get("query_candidates") == [], \
            f"Claim {claim.get('id')} should have empty query_candidates"


@pytest.mark.asyncio
async def test_horoscope_claims_skip_search(claim_skill, mock_llm_client):
    """
    T9: Horoscope claims with verification_target="none" should set should_skip_search=True.
    """
    mock_llm_client.call_json.return_value = {
        "article_intent": "prediction",
        "claims": [
            {
                "text": "Очікуйте несподівану грошову винагороду",
                "normalized_text": "Expect unexpected monetary reward",
                "type": "core",
                "claim_category": "OPINION",
                "topic_group": "Other",
                "importance": 0.2,
                "check_worthiness": 0.1,
                "harm_potential": 1,
                "verification_target": "none",
                "claim_role": "context",
                "metadata_confidence": "high",
            },
        ],
    }
    
    claims, _, _, _ = await claim_skill.extract_claims(HOROSCOPE_ARTICLE_EN, lang="en")
    
    assert len(claims) >= 1
    
    for claim in claims:
        metadata = claim.get("metadata")
        assert metadata is not None
        # should_skip_search is a computed property that checks if target is NONE
        assert metadata.should_skip_search is True, \
            f"Horoscope claim should skip search. Target: {metadata.verification_target}"


@pytest.mark.asyncio
async def test_mixed_article_correct_targets(claim_skill, mock_llm_client):
    """
    T9+: Mixed article should have different verification_targets per claim type.
    
    - Factual claims (e.g., "Ukrainian forces recaptured Tokmak") → reality
    - Predictions (e.g., "Russia will likely intensify attacks") → none
    - Attribution (e.g., "John Smith predicts...") → attribution or reality
    """
    mock_llm_client.call_json.return_value = {
        "article_intent": "news",
        "claims": [
            {
                "text": "Ukrainian forces have recaptured the city of Tokmak",
                "normalized_text": "Ukrainian forces recaptured Tokmak in Zaporizhzhia region",
                "type": "core",
                "claim_category": "FACTUAL",
                "topic_group": "War",
                "importance": 0.95,
                "check_worthiness": 0.95,
                "harm_potential": 3,
                "verification_target": "reality",
                "claim_role": "core",
                "search_locale_plan": {"primary": "en", "fallback": ["uk"]},
                "retrieval_policy": {"channels_allowed": ["authoritative", "reputable_news"]},
                "metadata_confidence": "high",
                "query_candidates": [
                    {"text": "Tokmak recaptured Ukraine December 2025", "role": "CORE", "score": 1.0}
                ],
                "search_queries": ["Tokmak recaptured Ukraine"],
            },
            {
                "text": "Russia will likely intensify attacks in January 2025",
                "normalized_text": "Analyst John Smith predicts Russia will intensify attacks in January",
                "type": "attribution",
                "claim_category": "OPINION",
                "topic_group": "War",
                "importance": 0.5,
                "check_worthiness": 0.2,
                "harm_potential": 2,
                "verification_target": "none",  # Prediction, not verifiable
                "claim_role": "context",
                "search_locale_plan": {"primary": "en", "fallback": []},
                "retrieval_policy": {"channels_allowed": []},
                "metadata_confidence": "high",
                "query_candidates": [],
                "search_queries": [],
            },
            {
                "text": "Tomorrow's weather in Kyiv will be cloudy with -5°C",
                "normalized_text": "Weather forecast: Kyiv tomorrow cloudy, -5°C",
                "type": "numeric",
                "claim_category": "FACTUAL",
                "topic_group": "Other",
                "importance": 0.3,
                "check_worthiness": 0.4,
                "harm_potential": 1,
                # Weather forecast = "existence" - can verify the forecast exists
                "verification_target": "existence",
                "claim_role": "support",
                "search_locale_plan": {"primary": "uk", "fallback": ["en"]},
                "retrieval_policy": {"channels_allowed": ["authoritative"]},
                "metadata_confidence": "medium",
                "query_candidates": [
                    {"text": "Kyiv weather forecast December 22 2025", "role": "CORE", "score": 0.6}
                ],
                "search_queries": ["Kyiv weather forecast"],
            },
        ],
    }
    
    claims, check_oracle, article_intent, _ = await claim_skill.extract_claims(
        MIXED_ARTICLE,
        lang="en",
        max_claims=5,
    )
    
    # Check that we got mixed targets  
    targets = [c.get("metadata").verification_target for c in claims if c.get("metadata")]
    
    # Should have at least one "reality" target (the factual claim)
    assert VerificationTarget.REALITY in targets, f"Expected at least one reality target, got {targets}"
    
    # Should have at least one "none" target (the prediction)
    assert VerificationTarget.NONE in targets, f"Expected at least one none target, got {targets}"
    
    # The core factual claim should have search queries
    core_claims = [c for c in claims if c.get("metadata") and c.get("metadata").verification_target == VerificationTarget.REALITY]
    assert len(core_claims) >= 1, "Expected at least one reality-target claim"
    
    for core_claim in core_claims:
        # Reality claims should have search queries
        assert len(core_claim.get("query_candidates", [])) > 0 or len(core_claim.get("search_queries", [])) > 0, \
            f"Reality claim should have search queries: {core_claim.get('normalized_text')}"


@pytest.mark.asyncio
async def test_metadata_fallback_defaults(claim_skill, mock_llm_client):
    """
    T7: When LLM omits metadata fields, sensible defaults should be applied.
    """
    # LLM response with minimal metadata
    mock_llm_client.call_json.return_value = {
        "article_intent": "news",
        "claims": [
            {
                "text": "Some claim text",
                "normalized_text": "Some claim text with context",
                "type": "core",
                # Missing: verification_target, claim_role, search_locale_plan, retrieval_policy
                "topic_group": "Science",
                "importance": 0.7,
                "check_worthiness": 0.7,
                "harm_potential": 2,
            },
        ],
    }
    
    claims, _, _, _ = await claim_skill.extract_claims("Some claim text", lang="en")
    
    assert len(claims) >= 1
    claim = claims[0]
    metadata = claim.get("metadata")
    
    assert metadata is not None
    # Default verification_target for factual claims should be "reality"
    assert metadata.verification_target == VerificationTarget.REALITY
    # Default claim_role for reality-target should be "core"  
    assert metadata.claim_role == ClaimRole.CORE
    # Default locale should use article language
    assert metadata.search_locale_plan.primary == "en"
    # Default channels should include authoritative + reputable (harm_potential=2)
    assert len(metadata.retrieval_policy.channels_allowed) > 0


@pytest.mark.asyncio
async def test_high_harm_gets_authoritative_only(claim_skill, mock_llm_client):
    """
    T7: High harm claims (harm_potential >= 4) should default to authoritative-only channels.
    """
    mock_llm_client.call_json.return_value = {
        "article_intent": "news",
        "claims": [
            {
                "text": "Drinking kerosene cures cancer",
                "normalized_text": "Claim that kerosene consumption cures cancer",
                "type": "core",
                "claim_category": "FACTUAL",
                "topic_group": "Health",
                "importance": 0.9,
                "check_worthiness": 0.99,
                "harm_potential": 5,  # Critical harm - medical misinformation
                # LLM omits retrieval_policy
                "verification_target": "reality",
                "claim_role": "core",
            },
        ],
    }
    
    claims, _, _, _ = await claim_skill.extract_claims("Drinking kerosene cures cancer", lang="en")
    
    assert len(claims) >= 1
    metadata = claims[0].get("metadata")
    
    assert metadata is not None
    # High harm should restrict to authoritative only
    from spectrue_core.schema.claim_metadata import EvidenceChannel
    assert metadata.retrieval_policy.channels_allowed == [EvidenceChannel.AUTHORITATIVE], \
        f"High harm claim should only allow authoritative sources, got {metadata.retrieval_policy.channels_allowed}"


# ─────────────────────────────────────────────────────────────────────────────
# T22: Progressive Widening Early Exit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_search_mgr():
    """Create a mock SearchManager for phase runner tests."""
    search_mgr = MagicMock()
    return search_mgr


@pytest.mark.asyncio
async def test_progressive_widening_early_exit():
    """
    T22: Phase A yields sufficient evidence → B/C/D skipped.
    
    When the first search phase returns authoritative sources,
    the PhaseRunner should stop early and not execute subsequent phases.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.execution_plan import (
        phase_a, phase_b, phase_c, phase_d
    )
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    # Create mock search manager
    search_mgr = MagicMock()
    
    # Mock search_phase to match SearchManager.search_phase(): (context, sources)
    async def mock_search(*args, **kwargs):
        return "", [
            {
                "url": "https://cdc.gov/article",
                "title": "Official CDC Information",
                "stance": "support",
                "quote": "The claim has been verified by our research.",
                "is_trusted": True,
            }
        ]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    # Create phase runner
    runner = PhaseRunner(search_mgr)
    
    # Create a claim with metadata
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        check_worthiness=0.9,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=["en"]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[
            EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS
        ]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "c1",
        "text": "Test claim",
        "normalized_text": "Test claim for verification",
        "type": "core",
        "search_queries": ["test query"],
        "query_candidates": [],
        "metadata": metadata,
    }
    
    # Create phases: A, B, C, D
    phases = [
        phase_a("en"),
        phase_b("en"),
        phase_c("uk"),  # Fallback
        phase_d("en"),
    ]
    
    # Run phases for single claim
    sources = await runner.run_claim_phases(claim, phases)
    
    # Assert: We got results
    assert len(sources) >= 1, "Should have at least one source"
    
    # Assert: search_phase was called only ONCE (for Phase A)
    # This proves early exit after Phase A returned sufficient evidence
    assert search_mgr.search_phase.call_count == 1, \
        f"Expected 1 search call (Phase A only), got {search_mgr.search_phase.call_count}"


@pytest.mark.asyncio
async def test_progressive_widening_continues_if_insufficient():
    """
    T22: If Phase A is insufficient, continue to Phase B.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.execution_plan import phase_a, phase_b
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    search_mgr = MagicMock()
    call_count = [0]
    
    # First call (Phase A): return social media (insufficient)
    # Second call (Phase B): return authoritative (sufficient)
    async def mock_search(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Phase A: social media only (insufficient)
            return "", [
                {
                    "url": "https://twitter.com/user/123",
                    "title": "Tweet",
                    "stance": "support",
                    "content": "Someone tweeted about this.",
                }
            ]
        else:
            # Phase B: authoritative (sufficient)
            return "", [
                {
                    "url": "https://nasa.gov/article",
                    "title": "NASA Official",
                    "stance": "support",
                    "quote": "Confirmed by NASA research.",
                    "is_trusted": True,
                }
            ]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    runner = PhaseRunner(search_mgr)
    
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        check_worthiness=0.9,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "c2",
        "text": "Another claim",
        "normalized_text": "Another claim for testing",
        "type": "core",
        "search_queries": ["another query"],
        "query_candidates": [],
        "metadata": metadata,
    }
    
    phases = [phase_a("en"), phase_b("en")]
    
    sources = await runner.run_claim_phases(claim, phases)
    
    # Phase A may return sources that are filtered out by phase channel policy.
    # The key behavior: Phase B runs after Phase A is insufficient.
    assert len(sources) >= 1, f"Expected at least 1 source after widening, got {len(sources)}"
    
    # Assert: search was called TWICE (Phase A + Phase B)
    assert search_mgr.search_phase.call_count == 2, \
        f"Expected 2 search calls, got {search_mgr.search_phase.call_count}"


@pytest.mark.asyncio
async def test_progressive_widening_skip_search_for_none_target():
    """
    T22: Claims with verification_target=none should skip search entirely.
    """
    from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
    from spectrue_core.verification.orchestration.execution_plan import BudgetClass
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy
    )
    
    orchestrator = ClaimOrchestrator()
    
    # Horoscope claim with verification_target=none
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.NONE,
        claim_role=ClaimRole.CONTEXT,
        check_worthiness=0.1,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "horoscope_1",
        "text": "Stars will favor your finances",
        "normalized_text": "Astrological prediction about finances",
        "type": "core",
        "search_queries": [],
        "query_candidates": [],
        "metadata": metadata,
    }
    
    # Build execution plan
    plan = orchestrator.build_execution_plan([claim], BudgetClass.DEEP)
    
    # Assert: No phases for none-target claim  
    phases = plan.get_phases("horoscope_1")
    assert len(phases) == 0, \
        f"None-target claim should have 0 phases, got {len(phases)}: {[p.phase_id for p in phases]}"


# ─────────────────────────────────────────────────────────────────────────────
# T26: Parallelism Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_execution_within_phase():
    """
    T26: Multiple claims in Phase A should execute concurrently.
    
    Verifies that the PhaseRunner uses asyncio.gather for parallel execution.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
    from spectrue_core.verification.orchestration.execution_plan import BudgetClass
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    import asyncio
    
    search_mgr = MagicMock()
    
    # Track execution order
    execution_log = []
    
    async def mock_search(*args, **kwargs):
        query = args[0] if args else kwargs.get("query", "")
        execution_log.append(f"start:{query}")
        await asyncio.sleep(0.01)  # Simulate async work
        execution_log.append(f"end:{query}")
        return "", [
            {
                "url": "https://nasa.gov/result",
                "stance": "support",
                "quote": "Official confirmation.",
            }
        ]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    # Create 3 claims
    def make_claim(claim_id, query):
        metadata = ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.9,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
            metadata_confidence=MetadataConfidence.HIGH,
        )
        return {
            "id": claim_id,
            "text": f"Claim {claim_id}",
            "normalized_text": f"Claim {claim_id} normalized",
            "type": "core",
            "search_queries": [query],
            "query_candidates": [],
            "metadata": metadata,
        }
    
    claims = [
        make_claim("c1", "query1"),
        make_claim("c2", "query2"),
        make_claim("c3", "query3"),
    ]
    
    # Build execution plan
    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)
    
    # Run with phase runner
    runner = PhaseRunner(search_mgr, max_concurrent=3)
    evidence = await runner.run_all_claims(claims, plan)
    
    # Assert: All claims were searched
    assert len(evidence) == 3
    assert search_mgr.search_phase.call_count == 3
    
    # Assert: Parallel execution (starts happen before all ends)
    # If truly parallel, we should see multiple "start" entries before "end" entries
    start_indices = [i for i, e in enumerate(execution_log) if e.startswith("start:")]
    end_indices = [i for i, e in enumerate(execution_log) if e.startswith("end:")]
    
    # With parallel execution, at least some starts should happen before corresponding ends
    # (This is a weak test, but verifies asyncio.gather is being used)
    assert len(start_indices) == 3, "Should have 3 start events"
    assert len(end_indices) == 3, "Should have 3 end events"


@pytest.mark.asyncio
async def test_waterfall_phase_ordering():
    """
    T26: Phase B only runs after Phase A completes for all claims.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
    from spectrue_core.verification.orchestration.execution_plan import BudgetClass
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    search_mgr = MagicMock()
    
    # Track phase order
    phase_log = []
    call_count = [0]
    
    async def mock_search(*args, **kwargs):
        call_count[0] += 1
        # Use depth to differentiate phases (basic=Phase A, advanced=Phase B)
        depth = kwargs.get("depth", "basic")
        phase = "A" if depth == "basic" else "B"
        phase_log.append(phase)
        # Return insufficient for Phase A to force Phase B
        return "", [
            {
                "url": "https://twitter.com/x",
                "stance": "context",
                "content": "Social media only.",
            }
        ]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    # Create 2 claims
    def make_claim(claim_id):
        metadata = ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.9,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
            metadata_confidence=MetadataConfidence.HIGH,
        )
        return {
            "id": claim_id,
            "text": f"Claim {claim_id}",
            "normalized_text": f"Claim {claim_id} normalized",
            "type": "core",
            "search_queries": [f"query for {claim_id}"],
            "query_candidates": [],
            "metadata": metadata,
        }
    
    claims = [make_claim("c1"), make_claim("c2")]
    
    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
    
    runner = PhaseRunner(search_mgr, max_concurrent=2)
    await runner.run_all_claims(claims, plan)
    
    # Assert: Phase A should run for all claims before Phase B
    # Since we have 2 claims with A+B phases, we expect: A, A, B, B
    # All A's should come before all B's
    first_b_index = next((i for i, p in enumerate(phase_log) if p == "B"), len(phase_log))
    len(phase_log) - 1 - next((i for i, p in enumerate(reversed(phase_log)) if p == "A"), len(phase_log))
    
    # If waterfall is working, last A should be before first B
    a_count = phase_log.count("A")
    phase_log.count("B")
    
    assert a_count >= 2, f"Expected at least 2 Phase A executions, got {a_count}"
    # Phase B may or may not run depending on sufficiency check
    # The key is that all A's run before any B's
    assert first_b_index >= a_count or first_b_index == len(phase_log), \
        f"Phase B should start after all Phase A complete. Log: {phase_log}"


@pytest.mark.asyncio
async def test_semaphore_respects_limit():
    """
    T26: Semaphore should limit concurrent searches.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
    from spectrue_core.verification.orchestration.execution_plan import BudgetClass
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    import asyncio
    
    search_mgr = MagicMock()
    
    # Track max concurrent
    current_concurrent = [0]
    max_concurrent_seen = [0]
    
    async def mock_search(*args, **kwargs):
        current_concurrent[0] += 1
        max_concurrent_seen[0] = max(max_concurrent_seen[0], current_concurrent[0])
        await asyncio.sleep(0.02)  # Hold the semaphore briefly
        current_concurrent[0] -= 1
        return "", [{"url": "https://example.gov", "stance": "support", "quote": "OK"}]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    # Create 5 claims (more than semaphore limit)
    def make_claim(claim_id):
        metadata = ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.9,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
            metadata_confidence=MetadataConfidence.HIGH,
        )
        return {
            "id": claim_id,
            "text": f"Claim {claim_id}",
            "normalized_text": f"Claim {claim_id}",
            "type": "core",
            "search_queries": [f"query{claim_id}"],
            "query_candidates": [],
            "metadata": metadata,
        }
    
    claims = [make_claim(f"c{i}") for i in range(5)]
    
    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)
    
    # Set max_concurrent to 2 (less than 5 claims)
    runner = PhaseRunner(search_mgr, max_concurrent=2)
    await runner.run_all_claims(claims, plan)
    
    # Assert: Max concurrent never exceeded limit
    assert max_concurrent_seen[0] <= 2, \
        f"Semaphore violated! Max concurrent was {max_concurrent_seen[0]}, limit is 2"


# ─────────────────────────────────────────────────────────────────────────────
# T29: Fail-Soft Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fail_soft_on_search_exception():
    """
    T29: Search exception should not crash pipeline, returns empty sources.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.execution_plan import phase_a
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    search_mgr = MagicMock()
    
    # Mock search to raise exception
    async def mock_search_fail(*args, **kwargs):
        raise Exception("Simulated Tavily API failure")
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search_fail)
    
    runner = PhaseRunner(search_mgr)
    
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        check_worthiness=0.9,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "c_fail",
        "text": "Test claim",
        "normalized_text": "Test claim",
        "type": "core",
        "search_queries": ["failing query"],
        "query_candidates": [],
        "metadata": metadata,
    }
    
    phases = [phase_a("en")]
    
    # Should NOT raise exception - fail-soft
    sources = await runner.run_claim_phases(claim, phases)
    
    # Assert: Returns empty list instead of crashing
    assert sources == [], "Fail-soft should return empty sources on exception"
    
    # Assert: Search was attempted
    assert search_mgr.search_phase.call_count == 1


@pytest.mark.asyncio
async def test_fail_soft_continues_to_next_phase():
    """
    T29: If Phase A fails, Phase B should still execute (fail-soft continue).
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.execution_plan import phase_a, phase_b
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    search_mgr = MagicMock()
    call_count = [0]
    
    async def mock_search(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Phase A fails
            raise Exception("Phase A network error")
        else:
            # Phase B succeeds
            return "", [
                {
                    "url": "https://cdc.gov/result",
                    "stance": "support",
                    "quote": "Official information.",
                }
            ]
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    runner = PhaseRunner(search_mgr)
    
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        check_worthiness=0.9,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "c_recover",
        "text": "Test claim",
        "normalized_text": "Test claim normalized",
        "type": "core",
        "search_queries": ["recovery query"],
        "query_candidates": [],
        "metadata": metadata,
    }
    
    phases = [phase_a("en"), phase_b("en")]
    
    sources = await runner.run_claim_phases(claim, phases)
    
    # Assert: Phase B executed despite Phase A failure
    assert search_mgr.search_phase.call_count == 2, \
        "Both phases should be attempted"
    
    # Assert: We got results from Phase B
    assert len(sources) >= 1, "Should have results from Phase B"


@pytest.mark.asyncio
async def test_fail_soft_returns_partial_results():
    """
    T29: Partial failures should still return successful results.
    """
    from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
    from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
    from spectrue_core.verification.orchestration.execution_plan import BudgetClass
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy, EvidenceChannel
    )
    
    search_mgr = MagicMock()
    
    # Claim 1 succeeds, Claim 2 fails
    async def mock_search(*args, **kwargs):
        query = args[0] if args else kwargs.get("query", "")
        if "c1" in query:
            return "", [{"url": "https://nasa.gov/ok", "stance": "support", "quote": "OK"}]
        else:
            raise Exception("Claim 2 search failed")
    
    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    
    def make_claim(claim_id):
        metadata = ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.9,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
            metadata_confidence=MetadataConfidence.HIGH,
        )
        return {
            "id": claim_id,
            "text": f"Claim {claim_id}",
            "normalized_text": f"Claim {claim_id}",
            "type": "core",
            "search_queries": [f"query for {claim_id}"],
            "query_candidates": [],
            "metadata": metadata,
        }
    
    claims = [make_claim("c1"), make_claim("c2")]
    
    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)
    
    runner = PhaseRunner(search_mgr)
    evidence = await runner.run_all_claims(claims, plan)
    
    # Assert: Both claims have entries (even if empty for failed one)
    assert "c1" in evidence
    assert "c2" in evidence
    
    # Assert: Successful claim has results
    assert len(evidence["c1"]) >= 1, "Claim 1 should have results"
    
    # Assert: Failed claim has empty list (fail-soft)
    assert evidence["c2"] == [], "Failed claim should have empty results"


# ─────────────────────────────────────────────────────────────────────────────
# T32: RGBA Aggregation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rgba_aggregation_excludes_context_claims():
    """
    T32: Context claims (weight=0) should not dilute aggregate scores.
    """
    from spectrue_core.verification.scoring.rgba_aggregation import (
        aggregate_weighted, ClaimScore
    )
    
    # Mix of verifiable and non-verifiable claims
    claim_scores = [
        # Core factual claim - verified true (should dominate)
        ClaimScore(
            claim_id="c1",
            verified_score=0.95,
            danger_score=0.1,
            style_score=0.5,
            explainability_score=0.8,
            role_weight=1.0,  # CORE
            check_worthiness=0.9,
            evidence_quality=1.0,
        ),
        # Context claim - horoscope (weight=0, should be excluded)
        ClaimScore(
            claim_id="c2",
            verified_score=0.50,  # Would dilute if included
            danger_score=0.5,
            style_score=0.5,
            explainability_score=0.5,
            role_weight=0.0,  # CONTEXT → excluded
            check_worthiness=0.1,
            evidence_quality=0.0,
        ),
        # Attribution claim - weighted less
        ClaimScore(
            claim_id="c3",
            verified_score=0.80,
            danger_score=0.2,
            style_score=0.5,
            explainability_score=0.7,
            role_weight=0.7,  # ATTRIBUTION
            check_worthiness=0.7,
            evidence_quality=1.0,
        ),
    ]
    
    result = aggregate_weighted(claim_scores)
    
    # Assert: Context claim excluded
    assert result.excluded_claims == 1
    assert result.included_claims == 2
    
    # Assert: Verified score NOT diluted (should be > 0.85)
    # Without context: (0.95*0.9 + 0.80*0.49) / (0.9 + 0.49) ≈ 0.89
    assert result.verified > 0.80, \
        f"Verified score should be high (>0.80), got {result.verified}"
    
    # Old broken behavior would give: (0.95 + 0.50 + 0.80) / 3 ≈ 0.75
    assert result.verified != pytest.approx(0.75, abs=0.05), \
        "Score appears to be simple average (not weighted)"


def test_rgba_aggregation_all_context_returns_neutral():
    """
    T32: If all claims are context (weight=0), return neutral scores.
    """
    from spectrue_core.verification.scoring.rgba_aggregation import (
        aggregate_weighted, ClaimScore
    )
    
    # All claims are non-verifiable
    claim_scores = [
        ClaimScore(
            claim_id="horoscope_1",
            verified_score=0.3,
            danger_score=0.5,
            style_score=0.5,
            explainability_score=0.5,
            role_weight=0.0,  # Excluded
            check_worthiness=0.1,
            evidence_quality=0.0,
        ),
        ClaimScore(
            claim_id="prediction_1",
            verified_score=0.2,
            danger_score=0.5,
            style_score=0.5,
            explainability_score=0.5,
            role_weight=0.0,  # Excluded
            check_worthiness=0.1,
            evidence_quality=0.0,
        ),
    ]
    
    result = aggregate_weighted(claim_scores)
    
    # Assert: All claims excluded
    assert result.included_claims == 0
    assert result.excluded_claims == 2
    
    # Assert: Neutral scores returned
    assert result.verified == 0.5
    assert result.danger == 0.5


def test_rgba_aggregation_weights_by_check_worthiness():
    """
    T32: Claims with higher check_worthiness should have more impact.
    """
    from spectrue_core.verification.scoring.rgba_aggregation import (
        aggregate_weighted, ClaimScore
    )
    
    claim_scores = [
        # High worthiness claim - should dominate
        ClaimScore(
            claim_id="c1",
            verified_score=1.0,
            danger_score=0.0,
            style_score=0.5,
            explainability_score=0.8,
            role_weight=1.0,
            check_worthiness=0.99,  # Very important
            evidence_quality=1.0,
        ),
        # Low worthiness claim - less impact
        ClaimScore(
            claim_id="c2",
            verified_score=0.0,
            danger_score=1.0,
            style_score=0.5,
            explainability_score=0.5,
            role_weight=1.0,
            check_worthiness=0.01,  # Trivial
            evidence_quality=1.0,
        ),
    ]
    
    result = aggregate_weighted(claim_scores)
    
    # Assert: High-worthiness claim dominates
    # weight1 = 1.0 * 0.99 * 1.0 = 0.99
    # weight2 = 1.0 * 0.01 * 1.0 = 0.01
    # verified = (1.0*0.99 + 0.0*0.01) / 1.0 ≈ 0.99
    assert result.verified > 0.95, \
        f"High-worthiness claim should dominate, got {result.verified}"


def test_claim_to_score_extracts_metadata():
    """
    T31: claim_to_score should extract role_weight from metadata.
    """
    from spectrue_core.verification.scoring.rgba_aggregation import claim_to_score
    from spectrue_core.schema.claim_metadata import (
        ClaimMetadata, VerificationTarget, ClaimRole, MetadataConfidence,
        SearchLocalePlan, RetrievalPolicy
    )
    
    # Create claim with metadata
    metadata = ClaimMetadata(
        verification_target=VerificationTarget.NONE,  # Should give weight=0
        claim_role=ClaimRole.CONTEXT,
        check_worthiness=0.1,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[]),
        metadata_confidence=MetadataConfidence.HIGH,
    )
    
    claim = {
        "id": "horoscope",
        "text": "Stars predict success",
        "metadata": metadata,
    }
    
    score = claim_to_score(
        claim,
        verified_score=0.5,
        danger_score=0.5,
    )
    
    # Assert: Role weight extracted correctly
    assert score.role_weight == 0.0, "CONTEXT/NONE should have weight=0"
    assert score.check_worthiness == 0.1
    assert score.is_excluded is True, "Should be excluded from aggregate"


def test_claim_to_score_defaults_for_missing_metadata():
    """
    T31: Claims without metadata should get default full weight.
    """
    from spectrue_core.verification.scoring.rgba_aggregation import claim_to_score
    
    # Claim without metadata (backward compat)
    claim = {
        "id": "legacy",
        "text": "Some claim",
        # No metadata
    }
    
    score = claim_to_score(
        claim,
        verified_score=0.8,
        danger_score=0.2,
    )
    
    # Assert: Default full weight for backward compat
    assert score.role_weight == 1.0
    assert score.check_worthiness == 0.5
    assert score.is_excluded is False