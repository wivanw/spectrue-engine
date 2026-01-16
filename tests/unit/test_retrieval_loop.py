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
from spectrue_core.verification.orchestration.execution_plan import ExecutionPlan, phase_a
from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
from spectrue_core.verification.search.search_policy import SearchDepth, SearchPolicyProfile, SearchProfileName


@pytest.mark.asyncio
async def test_hop_limit_enforced_main_profile():
    search_mgr = MagicMock()
    search_mgr.search_phase = AsyncMock(
        return_value=("", [
            {"url": "https://example.com/a", "stance": "context", "snippet": "Example context evidence."},
        ])
    )
    # Mock apply_evidence_acquisition_ladder to avoid TypeError on await
    search_mgr.apply_evidence_acquisition_ladder = AsyncMock(side_effect=lambda x, **kwargs: x)


    profile = SearchPolicyProfile(
        name=SearchProfileName.GENERAL.value,
        max_hops=1,
        search_depth=SearchDepth.BASIC.value,
        max_results=3,
        channels_allowed=[EvidenceChannel.AUTHORITATIVE],
    )

    search_mgr.estimate_hop_cost = MagicMock(return_value=0.0)
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
    search_mgr.apply_evidence_acquisition_ladder = AsyncMock(side_effect=lambda x, **kwargs: x)


    profile = SearchPolicyProfile(
        name=SearchProfileName.DEEP.value,
        max_hops=2,
        search_depth=SearchDepth.ADVANCED.value,
        max_results=5,
        channels_allowed=[EvidenceChannel.REPUTABLE_NEWS],
    )

    search_mgr.estimate_hop_cost = MagicMock(return_value=0.0)
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
    assert not state.error, f"State error: {state.error}"
    assert state.sufficiency_reason == "followup_failed"
    assert search_mgr.search_phase.call_count == 1
