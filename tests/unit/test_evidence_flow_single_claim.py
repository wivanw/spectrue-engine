# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest

from spectrue_core.verification.pipeline.pipeline_evidence import run_evidence_flow, EvidenceFlowInput


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
async def test_normal_profile_allows_multiple_claims():
    agent = _DummyAgent()
    search_mgr = _DummySearchMgr()

    claims = [
        {"id": "c1", "text": "Перве твердження", "importance": 0.9},
        {"id": "c2", "text": "Друге твердження", "importance": 0.8},
    ]

    # Mock agent expects multiple claims now
    async def mock_score_evidence(pack, *, model="gpt-5.2", lang="en"):
        claims = pack.get("claims")
        assert len(claims) == 2
        return {
            "status": "ok",
            "verified_score": 0.5,
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.5},
                {"claim_id": "c2", "verdict_score": 0.5},
            ]
        }
    
    agent.score_evidence = mock_score_evidence

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
            
        ),
        claims=claims,
        sources=[],
    )

    assert result["status"] == "ok"
    # Normal mode allows multiple claims, so we get veridcts for both
    assert len(result["claim_verdicts"]) == 2




