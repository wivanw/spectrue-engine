import os

import pytest

from spectrue_core.verification.inline_verification import InlineVerificationSkill


class _DummyLLM:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)

    async def call_json(self, *args, **kwargs):  # noqa: ANN001
        if not self._responses:
            raise RuntimeError("No more canned responses")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_m67_promotes_by_expected_value_no_hard_threshold(monkeypatch):
    # Make false promotions costly but still allow promotion if probability is strong.
    monkeypatch.setenv("SPECTRUE_SOCIAL_BENEFIT_TRUE", "1.0")
    monkeypatch.setenv("SPECTRUE_SOCIAL_COST_FALSE", "2.0")

    llm = _DummyLLM(
        responses=[
            {"p": 0.95, "ci": [0.85, 0.99], "reasons": ["handle_match"], "rule_violations": []},
            {"p": 0.95, "ci": [0.85, 0.99], "reasons": ["explicit_quote"], "rule_violations": []},
        ]
    )
    skill = InlineVerificationSkill(config=None, llm_client=llm)
    out = await skill.verify_social_statement({"text": "X announced Y"}, "We announce Y", "https://x.com/X")
    assert out["tier"] == "A'"
    assert out["decision"]["ev"] > 0
    # Sanity: no more p>0.8 gating; we rely on EV and CI lower bound.
    assert out["decision"]["p_joint_low"] == pytest.approx(0.85 * 0.85, rel=1e-6)


@pytest.mark.asyncio
async def test_m67_does_not_promote_when_uncertain(monkeypatch):
    monkeypatch.setenv("SPECTRUE_SOCIAL_BENEFIT_TRUE", "1.0")
    monkeypatch.setenv("SPECTRUE_SOCIAL_COST_FALSE", "2.0")

    llm = _DummyLLM(
        responses=[
            {"p": 0.6, "ci": [0.2, 0.8], "reasons": ["insufficient_input"], "rule_violations": ["ambiguous_entity"]},
            {"p": 0.7, "ci": [0.3, 0.9], "reasons": ["related"], "rule_violations": []},
        ]
    )
    skill = InlineVerificationSkill(config=None, llm_client=llm)
    out = await skill.verify_social_statement({"text": "X announced Y"}, "Maybe Y", "")
    assert out["tier"] == "D"
    assert out["decision"]["ev"] <= 0
