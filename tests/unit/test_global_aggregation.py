# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Global G excludes claims that are not verifiable by nature (verification_target = none)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spectrue_core.adapters.llm.scoring import ScoringSkill
from spectrue_core.domain.claims.model.enums import VerificationTarget
from spectrue_core.domain.verification.verdict.global_aggregation import (
    aggregate_global_veracity,
    is_unverifiable_by_nature,
)


@pytest.mark.parametrize("claim,expected", [
    ({"verification_target": "none"}, True),
    ({"verification_target": VerificationTarget.NONE}, True),
    ({"metadata": {"verification_target": "none"}}, True),
    ({"metadata": SimpleNamespace(verification_target=VerificationTarget.NONE)}, True),
    ({"verification_target": "reality"}, False),
    ({}, False),
])
def test_is_unverifiable_by_nature(claim, expected):
    assert is_unverifiable_by_nature(claim) is expected


def test_unverifiable_claim_is_excluded_from_global_g():
    # The 2026-10-18 trace: c1 verified at 0.98, c2 "about 330 metres tall" has
    # verification_target none. Before: 0.98 * w1 / (w1 + w2) = 0.553.
    g, policy = aggregate_global_veracity({"c1": 0.98, "c2": -1.0}, {"c1": 0.8, "c2": 0.7}, {"c2"})
    assert g == pytest.approx(0.98)
    assert policy["excluded_unverifiable_by_nature"] == ["c2"]


def test_unverified_factual_claim_still_reduces_global_g():
    g, policy = aggregate_global_veracity({"c1": 0.9, "c2": -1.0}, {"c1": 1.0, "c2": 1.0}, set())
    assert g == pytest.approx(0.45)
    assert policy["policy"] == "importance_weighted_mean_capped_by_least_true"


def test_unverifiable_claim_cannot_be_the_anchor():
    # Highest importance claim is an opinion; anchor is the best verifiable one.
    g, policy = aggregate_global_veracity({"c1": -1.0, "c2": 0.7}, {"c1": 1.0, "c2": 0.5}, {"c1"})
    assert policy["anchor_claim_id"] == "c2" and g == pytest.approx(0.7)


def test_unverified_anchor_still_blocks_global():
    g, policy = aggregate_global_veracity({"c1": -1.0, "c2": 0.9}, {"c1": 1.0, "c2": 0.5}, set())
    assert g == -1.0 and policy["policy"] == "anchor_unverified_blocks_global"


def test_only_unverifiable_claims_gives_typed_status():
    g, policy = aggregate_global_veracity({"c1": 0.9}, {"c1": 1.0}, {"c1"})
    assert g == -1.0
    assert policy["status"] == "UNVERIFIABLE_BY_NATURE"


@pytest.mark.asyncio
async def test_score_evidence_marks_and_excludes_unverifiable_claim(mock_llm_client, mock_config):
    mock_config.runtime = MagicMock()
    mock_config.runtime.llm.timeout_sec = 30
    skill = ScoringSkill(config=mock_config, llm_client=mock_llm_client)
    mock_llm_client.call_json.return_value = {
        "claim_verdicts": [
            {"claim_id": "c1", "verdict_score": 0.96, "verdict": "verified", "reason": "r",
             "rgba": [0.02, 0.96, 0.96, 0.96], "prior_score": -1.0, "prior_reason": ""},
            {"claim_id": "c2", "verdict_score": 0.6, "verdict": "ambiguous", "reason": "r",
             "rgba": [0.03, 0.6, 0.95, 0.12], "prior_score": -1.0, "prior_reason": ""},
        ],
        "verified_score": 0.5, "explainability_score": 0.9, "danger_score": 0.02,
        "style_score": 0.96, "rationale": "x",
    }
    pack = {
        "claims": [
            {"id": "c1", "importance": 0.8, "verification_target": "reality"},
            {"id": "c2", "importance": 0.7, "verification_target": "none"},
        ],
        "search_results": [{"claim_id": "c1", "relevance_score": 0.9}, {"claim_id": "c2", "relevance_score": 0.9}],
        "scored_sources": [{"claim_id": "c1", "stance": "SUPPORT"}, {"claim_id": "c2", "stance": "SUPPORT"}],
    }
    result = await skill.score_evidence(pack)
    assert result["verified_score"] == pytest.approx(0.96)
    c2 = next(cv for cv in result["claim_verdicts"] if cv["claim_id"] == "c2")
    assert c2["status"] == "UNVERIFIABLE_BY_NATURE"
    assert c2["verdict_score"] == -1.0 and c2["rgba"][1] == -1.0


def test_global_g_is_capped_by_the_least_true_main_claim():
    """Six-fact run 2026-10-18: 'sign on a trash can' (0.94) + 'meant to make people bin ballots' (0.05) gave 0.49."""
    g, policy = aggregate_global_veracity({"c1": 0.94, "c2": 0.05}, {"c1": 0.9, "c2": 0.9}, set())
    assert g == pytest.approx(0.05) and policy["least_true"] == pytest.approx(0.05)


def test_context_claims_are_excluded_from_global_g():
    """'Delaware Democratic Senator Chris Coons' (context, 0.97) must not lift a refuted attribution (0.15)."""
    g, policy = aggregate_global_veracity({"c1": 0.15, "c3": 0.97}, {"c1": 0.9, "c3": 0.95}, set(), {"c3"})
    assert g == pytest.approx(0.15) and policy["excluded_context"] == ["c3"] and policy["anchor_claim_id"] == "c1"


def test_text_of_only_context_claims_is_still_judged():
    g, _ = aggregate_global_veracity({"c1": 0.8}, {"c1": 0.5}, set(), {"c1"})
    assert g == pytest.approx(0.8)


def test_is_context_claim_reads_role_from_claim_or_metadata():
    from spectrue_core.domain.verification.verdict.global_aggregation import is_context_claim
    assert is_context_claim({"claim_role": "context"}) and is_context_claim({"metadata": {"claim_role": "background"}})
    assert not is_context_claim({"claim_role": "core"}) and not is_context_claim({"claim_role": "attribution"})
