# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Judge verdict-vs-score consistency: one retry naming the contradiction, then unverified. No network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.adapters.llm.scoring import ScoringSkill
from spectrue_core.domain.verification.verdict.consistency import contradictions, label_score_contradiction
from spectrue_core.llm.model_registry import ModelID


@pytest.mark.parametrize("label, score, bad", [
    ("refuted", 0.99, True), ("unlikely", 0.7, True), ("verified", 0.2, True), ("REFUTE", 0.9, True),
    ("refuted", 0.03, False), ("verified", 0.95, False), ("ambiguous", 0.9, False), ("refuted", -1.0, False),
    ("unverified", 0.8, False), ("plausible", 0.65, False),
])
def test_label_score_contradiction(label, score, bad):
    assert (label_score_contradiction(label, score) is not None) is bad


def test_batch_contradictions_by_claim():
    found = contradictions([{"claim_id": "c1", "verdict": "refuted", "verdict_score": 0.99},
                            {"claim_id": "c2", "verdict": "verified", "verdict_score": 0.9}])
    assert list(found) == ["c1"]


def _response(score, verdict="refuted"):
    return {"claim_verdicts": [{"claim_id": "c1", "verdict_score": score, "verdict": verdict,
                                "reason": "Refuted by a direct NASA statement.", "rgba": [0.04, score, 0.95, 0.98],
                                "prior_score": -1.0, "prior_reason": "NEI"}],
            "verified_score": score, "explainability_score": 0.98, "danger_score": 0.04, "style_score": 0.95,
            "rationale": "Refuted by NASA."}


@pytest.fixture
def skill(mock_llm_client, mock_config):
    mock_config.openai_model = ModelID.PRO
    mock_config.runtime = MagicMock()
    mock_config.runtime.llm = MagicMock()
    mock_config.runtime.llm.timeout_sec = 30
    return ScoringSkill(config=mock_config, llm_client=mock_llm_client)


PACK = {"claims": [{"id": "c1", "importance": 1.0, "text": "The Great Wall is visible from the Moon"}],
        "search_results": [{"claim_id": "c1", "relevance_score": 0.9}],
        "scored_sources": [{"claim_id": "c1", "stance": "REFUTE"}]}


@pytest.mark.asyncio
async def test_contradiction_is_retried_and_the_consistent_answer_is_used(skill, mock_llm_client):
    """Production 2026-10-18: 'refuted by NASA' with score 0.99 showed 99% to the user."""
    mock_llm_client.call_json = AsyncMock(side_effect=[_response(0.99), _response(0.03)])
    result = await skill.score_evidence(PACK)
    assert mock_llm_client.call_json.call_count == 2
    assert "CONSISTENCY CHECK" in mock_llm_client.call_json.call_args_list[1].kwargs["input"]
    assert result["verified_score"] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_contradiction_twice_makes_the_claim_unverified(skill, mock_llm_client):
    mock_llm_client.call_json = AsyncMock(side_effect=[_response(0.99), _response(0.97)])
    result = await skill.score_evidence(PACK)
    assert result["verified_score"] == -1.0


@pytest.mark.asyncio
async def test_consistent_answer_is_not_retried(skill, mock_llm_client):
    mock_llm_client.call_json = AsyncMock(return_value=_response(0.03))
    result = await skill.score_evidence(PACK)
    assert mock_llm_client.call_json.call_count == 1 and result["verified_score"] == pytest.approx(0.03)
