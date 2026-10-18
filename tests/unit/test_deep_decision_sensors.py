# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147: edge typing and judge routing from the Decisions model. No network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.edge_sensor import RELATIONS, edge_questions, type_edges
from spectrue_core.adapters.llm.judge_model_routing import select_judge_model
from spectrue_core.adapters.llm.judge_routing_sensor import required_capability, routing_questions
from spectrue_core.domain.claims.graph.edges import CandidateEdge, EdgeRelation
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.llm.model_registry import ModelID
from spectrue_core.runtime_config import DecisionsConfig

LANGS = ("en", "uk", "ru", "de", "es", "fr", "ja", "zh")


def test_questions_in_every_locale():
    for lang in LANGS:
        assert [c["value"] for c in edge_questions(lang)[0].payload["choices"]] == list(RELATIONS)
        assert len(routing_questions(lang)[0].payload["levels"]) == 5
        assert "Prompt key" not in str(edge_questions(lang)[0].payload) + str(routing_questions(lang)[0].payload)


@pytest.mark.asyncio
async def test_edges_are_typed_one_request_each_with_probability_as_score():
    llm = MagicMock()
    llm.call_decision = AsyncMock(return_value=DecisionResult(
        answers={"relation": DecisionAnswer(name="relation", type="choice", choice="supports", confidence=0.83)},
        model="m", usage={}))
    edges = [CandidateEdge("c1", "c2", "sim", 0.8, True), CandidateEdge("c1", "c3", "adjacent", 0.5, False)]
    nodes = {c: SimpleNamespace(text=f"claim {c}") for c in ("c1", "c2", "c3")}
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        out = await type_edges(llm, DecisionsConfig(gate_modes={"edge_typing": "on"}), edges, nodes)
    assert llm.call_decision.call_count == 2
    assert out[0].relation == EdgeRelation.SUPPORTS and out[0].score == pytest.approx(0.83) and out[1].reason == "adjacent"


@pytest.mark.asyncio
@pytest.mark.parametrize("score, tier", [(0.0, "cheap"), (2.0, "mid"), (4.0, "high")])
async def test_difficulty_level_sets_the_judge_tier(score, tier):
    llm = MagicMock()
    llm.call_decision = AsyncMock(return_value=DecisionResult(
        answers={"difficulty": DecisionAnswer(name="difficulty", type="score", score=score, confidence=0.8)},
        model="m", usage={}))
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        got = await required_capability(llm, DecisionsConfig(gate_modes={"judge_routing": "on"}),
                                        claim_text="x", evidence=[{"stance": "support", "quote": "q"}], claim_id="c1")
    assert got == tier


@pytest.mark.asyncio
async def test_routing_off_leaves_the_rule_in_charge():
    llm = MagicMock(); llm.call_decision = AsyncMock()
    assert await required_capability(llm, DecisionsConfig(), claim_text="x", evidence=[], claim_id="c1") is None
    assert not llm.call_decision.called


def test_override_forces_the_tier():
    claim = SimpleNamespace(importance=0.9, check_worthiness=0.9, claim_role="core", harm_potential=5)
    d = select_judge_model(claim=claim, evidence_items=[], prompt_chars=2000, required_capability_override="cheap")
    assert d.required_capability == "cheap" and d.model in (ModelID.NANO, ModelID.MID, ModelID.PRO)
    assert select_judge_model(claim=claim, evidence_items=[], prompt_chars=2000).required_capability == "high"
