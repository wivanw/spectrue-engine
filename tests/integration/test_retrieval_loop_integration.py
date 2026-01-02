# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Integration test for two-hop retrieval loop.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    ClaimRole,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    EvidenceChannel,
)
from spectrue_core.verification.orchestration.execution_plan import ExecutionPlan, phase_a, phase_b
from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
from spectrue_core.verification.search.search_policy import SearchPolicyProfile


@pytest.mark.asyncio
async def test_two_hop_loop_reaches_sufficiency():
    search_mgr = MagicMock()
    call_count = {"n": 0}

    async def mock_search(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Use a reputable news domain (bbc.com) so it passes channel filtering
            # BBC is REPUTABLE_NEWS but NOT authoritative - won't satisfy Rule1
            return "", [
                {
                    "url": "https://bbc.com/first",
                    "stance": "support",
                    "snippet": "Initial report mentions the claim but lacks exact figures.",
                    "relevance_score": 0.9,
                }
            ]
        return "", [
            {
                "url": "https://cdc.gov/article",
                "stance": "support",
                "quote": "Official data confirms the claim.",
                "relevance_score": 0.9,
            }
        ]

    search_mgr.search_phase = AsyncMock(side_effect=mock_search)
    # Mock apply_evidence_acquisition_ladder to avoid TypeError on await
    search_mgr.apply_evidence_acquisition_ladder = AsyncMock(side_effect=lambda x: x)

    from spectrue_core.verification.search.search_policy import QualityThresholds

    profile = SearchPolicyProfile(
        name="deep",
        max_hops=3,
        search_depth="advanced",
        max_results=5,
        channels_allowed=[EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS],
        # Disable coverage check since mock sources don't have relevance_score
        quality_thresholds=QualityThresholds(min_coverage=0.0, min_diversity=0.0),
    )

    runner = PhaseRunner(search_mgr, use_retrieval_loop=True, policy_profile=profile)

    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ]),
        metadata_confidence=MetadataConfidence.HIGH,
    )

    claim = {
        "id": "c1",
        "text": "Test claim",
        "normalized_text": "Test claim",
        "search_queries": ["test query"],
        "metadata": metadata,
    }

    plan = ExecutionPlan()
    plan.add_claim("c1", [phase_a("en"), phase_b("en")])

    evidence = await runner.run_all_claims([claim], plan)

    assert search_mgr.search_phase.call_count == 2
    assert len(evidence.get("c1", [])) == 2
    assert runner.execution_state.get_or_create("c1").is_sufficient is True
