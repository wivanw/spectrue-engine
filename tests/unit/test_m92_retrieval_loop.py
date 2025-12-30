# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Unit tests for bounded retrieval loop.
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
from spectrue_core.verification.execution_plan import ExecutionPlan, phase_a
from spectrue_core.verification.phase_runner import PhaseRunner
from spectrue_core.verification.search_policy import SearchPolicyProfile


@pytest.mark.asyncio
async def test_hop_limit_enforced_main_profile():
    search_mgr = MagicMock()
    search_mgr.search_phase = AsyncMock(
        return_value=("", [
            {"url": "https://example.com/a", "stance": "context", "snippet": "Example context evidence."},
        ])
    )
    # Mock apply_evidence_acquisition_ladder to avoid TypeError on await
    search_mgr.apply_evidence_acquisition_ladder = AsyncMock(side_effect=lambda x: x)


    profile = SearchPolicyProfile(
        name="main",
        max_hops=1,
        search_depth="basic",
        max_results=3,
        channels_allowed=[EvidenceChannel.AUTHORITATIVE],
    )

    runner = PhaseRunner(search_mgr, use_retrieval_loop=True, policy_profile=profile)

    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
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
    plan.add_claim("c1", [phase_a("en")])

    await runner.run_all_claims([claim], plan)

    assert search_mgr.search_phase.call_count == 1


@pytest.mark.asyncio
async def test_stop_when_followup_query_fails():
    search_mgr = MagicMock()
    search_mgr.search_phase = AsyncMock(
        return_value=("", [
            {"url": "https://example.com/a", "stance": "support"},
        ])
    )
    # Mock apply_evidence_acquisition_ladder to avoid TypeError on await
    search_mgr.apply_evidence_acquisition_ladder = AsyncMock(side_effect=lambda x: x)


    profile = SearchPolicyProfile(
        name="deep",
        max_hops=2,
        search_depth="advanced",
        max_results=5,
        channels_allowed=[EvidenceChannel.REPUTABLE_NEWS],
    )

    runner = PhaseRunner(search_mgr, use_retrieval_loop=True, policy_profile=profile)

    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.CORE,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.REPUTABLE_NEWS]),
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
    plan.add_claim("c1", [phase_a("en")])

    await runner.run_all_claims([claim], plan)

    state = runner.execution_state.get_or_create("c1")
    assert state.sufficiency_reason == "followup_failed"
    assert search_mgr.search_phase.call_count == 1
