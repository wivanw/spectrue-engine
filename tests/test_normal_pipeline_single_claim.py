# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest

from spectrue_core.verification.pipeline_evidence import run_evidence_flow, EvidenceFlowInput


class _DummyAgent:
    async def cluster_evidence(self, claims, sources, stance_pass_mode="single"):
        return None

    async def score_evidence(self, pack, *, model="gpt-5.2", lang="en"):
        # If single-claim enforcement works, this should only ever see one claim in normal.
        assert isinstance(pack, dict)
        claims = pack.get("claims")
        assert isinstance(claims, list)
        assert len(claims) == 1
        return {
            "status": "ok",
            "verified_score": 0.5,
            "explainability_score": 0.5,
            "danger_score": 0.0,
            "style_score": 1.0,
            "claim_verdicts": [
                {
                    "claim_id": claims[0].get("id"),
                    "verdict_score": 0.5,
                    "verdict": "ambiguous",
                    "reason": "...",
                }
            ],
            "rationale": "...",
        }


class _DummySearchMgr:
    def calculate_cost(self, *_args, **_kwargs):
        return 0

    def get_search_meta(self):
        return {}


def _build_evidence_pack(*, fact, claims, sources, search_results_clustered, content_lang, article_context=None):
    # Minimal pack shape used by ScoringSkill/run_evidence_flow.
    return {
        "original_fact": fact,
        "claims": claims,
        "search_results": sources,
        "scored_sources": sources,
        "context_sources": [],
        "items": [],
    }


def _noop_enrich_sources_with_trust(sources):
    return sources


@pytest.mark.asyncio
async def test_normal_profile_enforces_single_claim():
    agent = _DummyAgent()
    search_mgr = _DummySearchMgr()

    claims = [
        {"id": "c1", "text": "Перве твердження", "importance": 0.9},
        {"id": "c2", "text": "Друге твердження", "importance": 0.8},
    ]

    result = await run_evidence_flow(
        agent=agent,
        search_mgr=search_mgr,
        build_evidence_pack=_build_evidence_pack,
        enrich_sources_with_trust=_noop_enrich_sources_with_trust,
        calibration_registry=None,
        inp=EvidenceFlowInput(
            fact="x",
            original_fact="x",
            lang="uk",
            content_lang="uk",
            gpt_model="gpt-5.2",
            search_type="smart",
            progress_callback=None,
            pipeline="normal",
        ),
        claims=claims,
        sources=[],
    )

    assert result["status"] == "ok"
    assert len(result["claim_verdicts"]) == 1


@pytest.mark.asyncio
async def test_normal_profile_raises_if_anchor_missing():
    agent = _DummyAgent()
    search_mgr = _DummySearchMgr()

    # Claims without ids -> anchor selection can't filter properly; should raise.
    claims = [{"text": "a", "importance": 1.0}, {"text": "b", "importance": 0.9}]

    with pytest.raises(RuntimeError):
        await run_evidence_flow(
            agent=agent,
            search_mgr=search_mgr,
            build_evidence_pack=_build_evidence_pack,
            enrich_sources_with_trust=_noop_enrich_sources_with_trust,
            calibration_registry=None,
            inp=EvidenceFlowInput(
                fact="x",
                original_fact="x",
                lang="uk",
                content_lang="uk",
                gpt_model="gpt-5.2",
                search_type="smart",
                progress_callback=None,
                pipeline="normal",
            ),
            claims=claims,
            sources=[],
        )

