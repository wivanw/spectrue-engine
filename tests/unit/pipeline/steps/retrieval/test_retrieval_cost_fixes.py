# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Deterministic retrieval cost fixes (no network, all providers mocked):

- B1: stage-2 re-search is gated by real per-claim sufficiency.
- B2: identical/fuzzy-duplicate queries are searched once per step.
- B3: skip-search claims get no per-claim queries in BuildQueriesStep.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.domain.claims.model import (
    ClaimMetadata,
    MetadataConfidence,
    VerificationTarget,
)
from spectrue_core.domain.claims.sufficiency import SufficiencyResult, SufficiencyStatus
from spectrue_core.pipeline.contracts import SEARCH_PLAN_KEY, SearchPlan
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import get_mode
from spectrue_core.pipeline.steps.retrieval.build_queries import BuildQueriesStep
from spectrue_core.pipeline.steps.retrieval.web_search import WebSearchStep

WEB_SEARCH = "spectrue_core.pipeline.steps.retrieval.web_search"


def _search_mgr() -> MagicMock:
    mgr = MagicMock()

    async def _search_phase(query, **_kwargs):
        slug = "-".join(query.lower().split())
        return "", [{"url": f"https://example.com/{slug}", "title": query, "content": query, "score": 0.9}]

    async def _fetch(urls, stage=None):
        return {u: f"text for {u}" for u in urls}

    mgr.search_phase = AsyncMock(side_effect=_search_phase)
    mgr.fetch_urls_content_batch = AsyncMock(side_effect=_fetch)
    return mgr


def _ctx(plan: SearchPlan, claims: list[dict], key_claim_ids: list[str]) -> PipelineContext:
    return (
        PipelineContext(mode=get_mode("general"), claims=claims)
        .set_extra(SEARCH_PLAN_KEY, plan)
        .set_extra("key_claim_ids", key_claim_ids)
    )


def _sufficiency(status: SufficiencyStatus):
    def _check(claim, sources):
        return SufficiencyResult(
            claim_id=str(claim.get("id")),
            status=status,
            posterior_p=0.95 if status == SufficiencyStatus.SUFFICIENT else 0.4,
        )

    return MagicMock(side_effect=_check)


def _searched(mgr: MagicMock) -> list[str]:
    return [c.args[0] for c in mgr.search_phase.call_args_list]


CLAIM = {"id": "c1", "text": "Claim one", "normalized_text": "Claim one"}


@pytest.mark.unit
class TestStage2SufficiencyGate:
    @pytest.mark.asyncio
    async def test_sufficient_claim_skips_stage2_search(self):
        plan = SearchPlan(plan_id="p1", mode="general", global_queries=("global topic",),
                          per_claim_queries={"c1": ("claim one evidence",)})
        mgr = _search_mgr()
        check = _sufficiency(SufficiencyStatus.SUFFICIENT)
        with patch(f"{WEB_SEARCH}.check_sufficiency_for_claim", check):
            out = await WebSearchStep(config=None, search_mgr=mgr, agent=None).run(_ctx(plan, [CLAIM], []))

        assert _searched(mgr) == ["global topic"]
        # Stage-2 decision is taken on the sources already bound to the claim.
        first_call_claim, first_call_sources = check.call_args_list[0].args
        assert first_call_claim["id"] == "c1"
        assert isinstance(first_call_sources, list)
        assert out.get_extra("cluster_sufficiency") == {"c1": 0.95}

    @pytest.mark.asyncio
    async def test_insufficient_claim_escalates_in_stage2(self):
        plan = SearchPlan(plan_id="p1", mode="general", global_queries=("global topic",),
                          per_claim_queries={"c1": ("claim one evidence",)})
        mgr = _search_mgr()
        with patch(f"{WEB_SEARCH}.check_sufficiency_for_claim", _sufficiency(SufficiencyStatus.INSUFFICIENT)):
            await WebSearchStep(config=None, search_mgr=mgr, agent=None).run(_ctx(plan, [CLAIM], []))

        assert _searched(mgr) == ["global topic", "claim one evidence"]

    @pytest.mark.asyncio
    async def test_skip_status_does_not_escalate(self):
        plan = SearchPlan(plan_id="p1", mode="general", global_queries=("global topic",),
                          per_claim_queries={"c1": ("claim one evidence",)})
        mgr = _search_mgr()
        with patch(f"{WEB_SEARCH}.check_sufficiency_for_claim", _sufficiency(SufficiencyStatus.SKIP)):
            await WebSearchStep(config=None, search_mgr=mgr, agent=None).run(_ctx(plan, [CLAIM], []))

        assert _searched(mgr) == ["global topic"]


@pytest.mark.unit
class TestCrossStageQueryDedup:
    @pytest.mark.asyncio
    async def test_stage2_repeat_of_stage1_query_is_not_searched_again(self):
        plan = SearchPlan(plan_id="p1", mode="general", global_queries=("global topic",),
                          per_claim_queries={"c1": ("claim one evidence",)})
        mgr = _search_mgr()
        trace = MagicMock()
        with (
            patch(f"{WEB_SEARCH}.check_sufficiency_for_claim", _sufficiency(SufficiencyStatus.INSUFFICIENT)),
            patch(f"{WEB_SEARCH}.Trace.event", trace),
        ):
            out = await WebSearchStep(config=None, search_mgr=mgr, agent=None).run(_ctx(plan, [CLAIM], ["c1"]))

        assert _searched(mgr) == ["global topic", "claim one evidence"]
        dup_events = [c.args[1] for c in trace.call_args_list
                      if c.args[0] == "retrieval.search.duplicate_query_skipped"]
        assert len(dup_events) == 1
        assert dup_events[0]["stage"] == 2
        assert dup_events[0]["claim_id"] == "c1"
        # The duplicate still contributes its (cached) URLs to the claim.
        assert out.sources

    @pytest.mark.asyncio
    async def test_per_claim_query_matching_global_query_is_deduped(self):
        plan = SearchPlan(plan_id="p1", mode="general", global_queries=("Claim One  Evidence",),
                          per_claim_queries={"c1": ("claim one evidence",)})
        mgr = _search_mgr()
        with patch(f"{WEB_SEARCH}.check_sufficiency_for_claim", _sufficiency(SufficiencyStatus.INSUFFICIENT)):
            await WebSearchStep(config=None, search_mgr=mgr, agent=None).run(_ctx(plan, [CLAIM], ["c1"]))

        assert mgr.search_phase.call_count == 1


@pytest.mark.unit
class TestBuildQueriesHonoursSkipSearch:
    @pytest.mark.asyncio
    async def test_skip_search_claim_gets_no_per_claim_queries(self):
        seed = ["alpha", "beta", "gamma", "delta"]
        claims = [
            {"id": "c1", "text": "Searchable claim", "retrieval_seed_terms": seed},
            {
                "id": "c2",
                "text": "Prediction claim",
                "retrieval_seed_terms": seed,
                "metadata": ClaimMetadata(verification_target=VerificationTarget.NONE),
            },
            {
                # M146 T020: no confidence fail-open; verification_target none is skipped.
                "id": "c3",
                "text": "Uncertain prediction",
                "retrieval_seed_terms": seed,
                "metadata": ClaimMetadata(
                    verification_target=VerificationTarget.NONE,
                    metadata_confidence=MetadataConfidence.LOW,
                ),
            },
            # Satire is mapped onto verification_target at parse time; a bare likelihood is not a skip path.
            {"id": "c4", "text": "Satire piece", "retrieval_seed_terms": seed, "satire_likelihood": 0.9},
        ]
        ctx = PipelineContext(mode=get_mode("general"), claims=claims)
        with patch(
            "spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan",
            return_value=["doc query"],
        ) as doc_plan:
            out = await BuildQueriesStep().run(ctx)

        plan: SearchPlan = out.get_extra(SEARCH_PLAN_KEY)
        assert plan.per_claim_queries["c1"]
        assert plan.per_claim_queries["c2"] == ()
        assert plan.per_claim_queries["c3"] == ()
        assert plan.per_claim_queries["c4"]
        planned_ids = [c["id"] for c in doc_plan.call_args.args[0]]
        assert planned_ids == ["c1", "c4"]

    @pytest.mark.asyncio
    async def test_all_claims_skipped_keeps_global_plan(self):
        claims = [{
            "id": "c1",
            "text": "Prediction claim",
            "metadata": ClaimMetadata(verification_target=VerificationTarget.NONE),
        }]
        ctx = PipelineContext(mode=get_mode("general"), claims=claims)
        with patch(
            "spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan",
            return_value=["doc query"],
        ):
            out = await BuildQueriesStep().run(ctx)

        plan: SearchPlan = out.get_extra(SEARCH_PLAN_KEY)
        assert plan.global_queries == ("doc query",)
        assert plan.per_claim_queries["c1"] == ()
