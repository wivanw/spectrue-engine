# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    ClaimRole,
    EvidenceChannel,
    MetadataConfidence,
    RetrievalPolicy,
    SearchLocalePlan,
    VerificationTarget,
)
from spectrue_core.verification.execution_plan import BudgetClass
from spectrue_core.verification.orchestrator import ClaimOrchestrator
from spectrue_core.verification.phase_runner import PhaseRunner


@pytest.mark.asyncio
async def test_phase_runner_dependency_ordering():
    """
    M93: Claims with dependencies should run after their premises.
    """
    search_mgr = MagicMock()
    execution_log: list[str] = []

    async def mock_search(*args, **kwargs):
        query = args[0] if args else kwargs.get("query", "")
        execution_log.append(query)
        await asyncio.sleep(0)
        return "", [{"url": "https://example.com"}]

    search_mgr.search_phase = AsyncMock(side_effect=mock_search)

    metadata = ClaimMetadata(
        verification_target=VerificationTarget.REALITY,
        claim_role=ClaimRole.THESIS,
        check_worthiness=0.9,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=[]),
        retrieval_policy=RetrievalPolicy(channels_allowed=[EvidenceChannel.AUTHORITATIVE]),
        metadata_confidence=MetadataConfidence.HIGH,
    )

    claims = [
        {
            "id": "c1",
            "text": "Premise claim",
            "normalized_text": "Premise claim",
            "type": "core",
            "search_queries": ["premise-query"],
            "query_candidates": [],
            "metadata": metadata,
            "structure": {
                "type": "event",
                "premises": [],
                "conclusion": "Premise claim",
                "dependencies": [],
            },
        },
        {
            "id": "c2",
            "text": "Conclusion claim",
            "normalized_text": "Conclusion claim",
            "type": "core",
            "search_queries": ["conclusion-query"],
            "query_candidates": [],
            "metadata": metadata,
            "structure": {
                "type": "causal",
                "premises": ["Premise claim"],
                "conclusion": "Conclusion claim",
                "dependencies": ["c1"],
            },
        },
    ]

    orchestrator = ClaimOrchestrator()
    plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)

    runner = PhaseRunner(search_mgr, max_concurrent=2)
    await runner.run_all_claims(claims, plan)

    assert execution_log == ["premise-query", "conclusion-query"]
    assert search_mgr.search_phase.call_count == 2
