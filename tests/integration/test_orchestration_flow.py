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
- T32: RGBA aggregation (context claims don't dilute scores)
"""

import pytest
from spectrue_core.runtime_config import DecisionsConfig
from unittest.mock import AsyncMock, MagicMock

from spectrue_core.adapters.llm.claims import ClaimExtractionSkill
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
    config.runtime.decisions = DecisionsConfig()  # M146: metadata sensor off
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
                # Validation fields
                "subject_entities": ["Aries", "Stars"],
                "retrieval_seed_terms": ["aries", "horoscope", "finance"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
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
                # Validation fields
                "subject_entities": ["Mercury", "Retrograde"],
                "retrieval_seed_terms": ["mercury", "retrograde", "finance"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
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
                # Validation fields
                "subject_entities": ["Reward"],
                "retrieval_seed_terms": ["unexpected", "monetary", "reward"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
            },
        ],
    }
    
    claims, _, _, _ = await claim_skill.extract_claims(HOROSCOPE_ARTICLE_EN, lang="en")
    
    assert len(claims) >= 1
    
    for claim in claims:
        metadata = claim.get("metadata")
        assert metadata is not None
        # should_skip_search is a computed property that checks if target is NONE
        from spectrue_core.domain.claims.policy import should_skip_search
        assert should_skip_search(metadata) is True, \
            f"Horoscope claim should skip search. Target: {metadata.verification_target}"


@pytest.mark.asyncio
async def test_mixed_article_correct_targets(claim_skill, mock_llm_client):
    """
    T9+: Mixed article should have different verification_targets per claim type.
    """
    # Define responses
    core_response = {
        "article_intent": "news",
        "claims": [
            {
                "text": "Ukrainian forces have recaptured the city of Tokmak",
                "normalized_text": "Ukrainian forces recaptured Tokmak in Zaporizhzhia region",
                "type": "core",
                "claim_category": "FACTUAL",
                "verification_target": "reality",
                # Validation
                "subject_entities": ["Tokmak", "Ukraine"],
                "retrieval_seed_terms": ["tokmak", "recaptured", "ukraine"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "event",
                 "time_anchor": {"type": "date", "date": "2025-12-01"},
            },
            {
                "text": "Russia will likely intensify attacks in January 2025",
                "normalized_text": "Analyst John Smith predicts Russia will intensify attacks in January",
                "type": "attribution",
                "claim_category": "OPINION",
                "verification_target": "none",
                # Validation
                "subject_entities": ["Russia"],
                "retrieval_seed_terms": ["russia", "attacks", "january"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
            },
            {
                "text": "Tomorrow's weather in Kyiv will be cloudy with -5°C",
                "normalized_text": "Weather forecast: Kyiv tomorrow cloudy, -5°C",
                "type": "numeric",
                "claim_category": "FACTUAL",
                "verification_target": "existence",
                # Validation
                "subject_entities": ["Kyiv"],
                "retrieval_seed_terms": ["kyiv", "weather", "forecast"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
            },
        ],
    }

    async def side_effect(*args, **kwargs):
        kind = kwargs.get("trace_kind")
        prompt = kwargs.get("input", "")
        
        if kind == "claim_extraction_core":
            return core_response
            
        if kind == "claim_retrieval_plan":
            # Return specific enrichment based on text in prompt
            # Use longer substrings to avoid matching context
            if "recaptured the city of Tokmak" in prompt:
                return {
                    "verification_target": "reality",
                     "claim_role": "core",
                     "search_queries": ["Tokmak query"],
                     "search_locale_plan": {"primary": "en", "fallback": []},
                     "retrieval_policy": {"channels_allowed": ["authoritative"]},
                     "metadata_confidence": "high",
                }
            elif "Russia will likely intensify" in prompt:
                return {
                    "verification_target": "none",
                    "claim_role": "context",
                    "search_queries": [],
                     "search_locale_plan": {"primary": "en", "fallback": []},
                     "retrieval_policy": {"channels_allowed": []},
                     "metadata_confidence": "high",
                }
            elif "Tomorrow's weather" in prompt:
                 return {
                    "verification_target": "existence",
                    "claim_role": "support",
                    "search_queries": ["Weather query"],
                     "search_locale_plan": {"primary": "uk", "fallback": []},
                     "retrieval_policy": {"channels_allowed": ["authoritative"]},
                     "metadata_confidence": "medium",
                }
            return {}
            
        return {}

    mock_llm_client.call_json.side_effect = side_effect
    
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
                # >= 20 chars: validate_core_claim rejects shorter text as
                # claim_text_too_short, which routed this through the synthetic
                # fallback claim (which carries no metadata) instead of the
                # real enrichment path this test is about.
                "text": "Some claim text that is long enough to validate",
                "normalized_text": "Some claim text with context",
                "type": "core",
                # Missing: verification_target, claim_role, search_locale_plan, retrieval_policy
                "topic_group": "Science",
                "importance": 0.7,
                "check_worthiness": 0.7,
                "harm_potential": 2,
                # Validation fields
                "subject_entities": ["Claim"],
                "retrieval_seed_terms": ["some", "claim", "text"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
            },
        ],
    }
    
    claims, _, _, _ = await claim_skill.extract_claims("Some claim text that is long enough to validate", lang="en")
    
    assert len(claims) >= 1
    claim = claims[0]
    metadata = claim.get("metadata")
    
    assert metadata is not None
    # Default verification_target for factual claims should be "reality"
    assert metadata.verification_target == VerificationTarget.REALITY
    # Default claim_role for reality-target should be "support"
    assert metadata.claim_role == ClaimRole.SUPPORT
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
                # Validation fields
                "subject_entities": ["Kerosene", "Cancer"],
                "retrieval_seed_terms": ["drinking", "kerosene", "cures", "cancer"],
                "falsifiability": {"is_falsifiable": True},
                "predicate_type": "existence",
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
# T32: RGBA Aggregation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rgba_aggregation_excludes_context_claims():
    """
    T32: Context claims (weight=0) should not dilute aggregate scores.
    """
    from spectrue_core.use_cases.verification.scoring.rgba_aggregation import (
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


def test_rgba_aggregation_all_context_returns_none():
    """
    T32: If all claims are context (weight=0), return None scores (not 0.5).
    """
    from spectrue_core.use_cases.verification.scoring.rgba_aggregation import (
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
    
    # Assert: None scores returned (not 0.5)
    assert result.verified is None
    assert result.danger is None


def test_rgba_aggregation_weights_by_check_worthiness():
    """
    T32: Claims with higher check_worthiness should have more impact.
    """
    from spectrue_core.use_cases.verification.scoring.rgba_aggregation import (
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
