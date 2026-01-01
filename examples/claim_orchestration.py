# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Claim Orchestration Example (M80)

This example demonstrates how to use the claim-centric orchestration
for fine-grained control over the verification process.
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock

from spectrue_core.verification.orchestrator import ClaimOrchestrator
from spectrue_core.verification.phase_runner import PhaseRunner
from spectrue_core.verification.execution_plan import BudgetClass
from spectrue_core.verification.sufficiency import evidence_sufficiency
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    ClaimRole,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    EvidenceChannel,
)


def create_sample_claims():
    """Create sample claims with different metadata."""
    
    # Claim 1: Factual claim (should be verified)
    factual_claim = {
        "id": "c1",
        "text": "The COVID-19 vaccine is 95% effective.",
        "normalized_text": "The COVID-19 vaccine has 95% effectiveness rate.",
        "type": "core",
        "importance": 0.9,
        "search_queries": ["COVID-19 vaccine effectiveness rate"],
        "query_candidates": [],
        "metadata": ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.95,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(
                channels_allowed=[EvidenceChannel.AUTHORITATIVE]
            ),
            metadata_confidence=MetadataConfidence.HIGH,
        ),
    }
    
    # Claim 2: Horoscope (should be skipped)
    horoscope_claim = {
        "id": "c2",
        "text": "Aries will have good luck this week.",
        "normalized_text": "Aries zodiac sign will experience good fortune.",
        "type": "prediction",
        "importance": 0.1,
        "search_queries": [],
        "query_candidates": [],
        "metadata": ClaimMetadata(
            verification_target=VerificationTarget.NONE,  # Not verifiable
            claim_role=ClaimRole.CONTEXT,
            check_worthiness=0.1,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[]),
            metadata_confidence=MetadataConfidence.HIGH,
        ),
    }
    
    # Claim 3: Attribution claim
    attribution_claim = {
        "id": "c3",
        "text": "Elon Musk said Tesla will release a new model next year.",
        "normalized_text": "Elon Musk announced Tesla new model release next year.",
        "type": "attribution",
        "importance": 0.7,
        "search_queries": ["Elon Musk Tesla new model announcement"],
        "query_candidates": [],
        "metadata": ClaimMetadata(
            verification_target=VerificationTarget.ATTRIBUTION,
            claim_role=ClaimRole.ATTRIBUTION,
            check_worthiness=0.8,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
            retrieval_policy=RetrievalPolicy(
                channels_allowed=[
                    EvidenceChannel.AUTHORITATIVE,
                    EvidenceChannel.REPUTABLE_NEWS,
                ]
            ),
            metadata_confidence=MetadataConfidence.MEDIUM,
        ),
    }
    
    return [factual_claim, horoscope_claim, attribution_claim]


async def main():
    print("=" * 60)
    print("CLAIM ORCHESTRATION EXAMPLE (M80)")
    print("=" * 60)
    
    # Create sample claims
    claims = create_sample_claims()
    print(f"\n📋 Created {len(claims)} sample claims")
    
    # Step 1: Build Execution Plan
    print("\n" + "-" * 40)
    print("Step 1: Build Execution Plan")
    print("-" * 40)
    
    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
    
    for claim in claims:
        claim_id = claim["id"]
        phases = plan.get_phases(claim_id)
        metadata = claim["metadata"]
        
        print(f"\n  Claim {claim_id}: {claim['text'][:50]}...")
        print(f"    Target: {metadata.verification_target.value}")
        print(f"    Role: {metadata.claim_role.value}")
        print(f"    Phases: {[p.phase_id for p in phases]}")
        
        if metadata.should_skip_search:
            print("    ⏭️  SKIP (not verifiable)")
    
    # Step 2: Simulate Phase Execution
    print("\n" + "-" * 40)
    print("Step 2: Simulate Phase Execution")
    print("-" * 40)
    
    # Create mock search manager
    search_mgr = MagicMock()
    
    async def mock_search(query, topic, intent):
        # Simulate finding authoritative source for factual claim
        if "COVID" in query:
            return [
                {
                    "url": "https://cdc.gov/vaccines/covid-19",
                    "title": "CDC COVID-19 Vaccine Information",
                    "content": "Studies show vaccine efficacy of 95%...",
                    "snippet": "The Pfizer vaccine showed 95% effectiveness.",
                    "stance": "support",
                    "quote": "Clinical trials demonstrated 95% efficacy.",
                    "is_trusted": True,
                }
            ]
        elif "Elon Musk" in query:
            return [
                {
                    "url": "https://reuters.com/tesla-announcement",
                    "title": "Tesla CEO announces new model",
                    "content": "Musk confirmed new model plans...",
                    "stance": "support",
                    "quote": "We will release a new model next year.",
                }
            ]
        return []
    
    search_mgr.search_unified = AsyncMock(side_effect=mock_search)
    
    # Run PhaseRunner
    runner = PhaseRunner(search_mgr, max_concurrent=3)
    evidence = await runner.run_all_claims(claims, plan)
    
    print("\n  Results:")
    for claim_id, sources in evidence.items():
        claim = next(c for c in claims if c["id"] == claim_id)
        print(f"\n  Claim {claim_id}:")
        print(f"    Sources found: {len(sources)}")
        
        if sources:
            # Check sufficiency
            metadata = claim["metadata"]
            sufficiency = evidence_sufficiency(
                claim_id=claim_id,
                sources=sources,
                verification_target=metadata.verification_target,
                claim_text=claim["text"],
            )
            print(f"    Sufficiency: {sufficiency.status.value}")
            if sufficiency.rule_matched:
                print(f"    Rule matched: {sufficiency.rule_matched}")
    
    # Step 3: RGBA Aggregation
    print("\n" + "-" * 40)
    print("Step 3: RGBA Aggregation")
    print("-" * 40)
    
    from spectrue_core.verification.rgba_aggregation import (
        aggregate_weighted,
        ClaimScore,
    )
    
    # Create scores (would come from scoring skill in real usage)
    scores = [
        ClaimScore(
            claim_id="c1",
            verified_score=0.95,  # High - CDC confirmed
            danger_score=0.1,
            style_score=0.8,
            explainability_score=0.9,
            role_weight=1.0,  # CORE
            check_worthiness=0.95,
            evidence_quality=1.0,
        ),
        ClaimScore(
            claim_id="c2",
            verified_score=0.5,  # Default
            danger_score=0.0,
            style_score=0.5,
            explainability_score=0.5,
            role_weight=0.0,  # CONTEXT - excluded!
            check_worthiness=0.1,
            evidence_quality=0.0,
        ),
        ClaimScore(
            claim_id="c3",
            verified_score=0.85,
            danger_score=0.2,
            style_score=0.7,
            explainability_score=0.8,
            role_weight=0.7,  # ATTRIBUTION
            check_worthiness=0.8,
            evidence_quality=1.0,
        ),
    ]
    
    result = aggregate_weighted(scores)
    
    print("\n  Aggregated RGBA (weighted):")
    print(f"    Verified:       {result.verified:.2f}")
    print(f"    Danger:         {result.danger:.2f}")
    print(f"    Style:          {result.style:.2f}")
    print(f"    Explainability: {result.explainability:.2f}")
    print("\n  Stats:")
    print(f"    Total claims:   {result.total_claims}")
    print(f"    Included:       {result.included_claims}")
    print(f"    Excluded:       {result.excluded_claims} (context/predictions)")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
