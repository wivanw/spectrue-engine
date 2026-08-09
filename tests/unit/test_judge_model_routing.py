"""Tests for the price-aware judge model router.

Regression context: `_estimate_credits_for_model` used to call three functions
that did not exist on the billing modules, so `select_judge_model` raised
AttributeError on every claim. Its only caller wraps it in a bare `except`
that falls back to the PRO tier, so the whole router was dead code and every
claim was judged by the most expensive model. The first test here is the guard
against that regression: it asserts a cheap claim does NOT end up on PRO.
"""

from types import SimpleNamespace

import pytest

from spectrue_core.adapters.llm.judge_model_routing import (
    UNPRICED_MODEL_CREDITS,
    _estimate_credits_for_model,
    select_judge_model,
)
from spectrue_core.llm.model_registry import ModelID


def _evidence(domain: str, *, stance: str = "support", tier: str = "A", quote: bool = False):
    return SimpleNamespace(
        domain=domain,
        stance=stance,
        source_reliability_hint=tier,
        has_quote=quote,
    )


def _simple_claim():
    """Low importance, low harm, low worthiness -> risk well under the 0.35 gate."""
    return SimpleNamespace(
        importance=0.3,
        check_worthiness=0.3,
        claim_role="support",
        harm_potential=0.5,
    )


def _clean_evidence():
    """Three high-tier domains, all agreeing -> no conflict, no unknown stance."""
    return [_evidence("apnews.com"), _evidence("reuters.com"), _evidence("bbc.co.uk")]


class TestSimpleClaimAvoidsProFallback:
    def test_simple_claim_does_not_route_to_pro(self):
        decision = select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )

        assert decision.model != ModelID.PRO, (
            "simple low-risk claim fell through to the PRO tier — the price-aware "
            "router is likely raising and being swallowed by the caller's fallback"
        )
        assert decision.required_capability == "cheap"
        assert decision.model == ModelID.NANO
        assert decision.reason == "cheap_ok_min_expected_cost"

    def test_selection_does_not_raise(self):
        """The caller swallows exceptions, so assert directly that none escape."""
        select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )

    def test_estimates_are_real_finite_costs(self):
        decision = select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )

        for tier in (ModelID.NANO, ModelID.MID, ModelID.PRO):
            cost = decision.est_credits[tier]
            assert isinstance(cost, float)
            assert 0.0 < cost < UNPRICED_MODEL_CREDITS, f"{tier} priced as {cost}"

        # PRO is the expensive tier; the router is only useful if it knows that.
        assert decision.est_credits[ModelID.PRO] > decision.est_credits[ModelID.NANO]

    def test_trace_payload_is_json_serializable(self):
        import json

        decision = select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )
        payload = json.loads(json.dumps(decision.to_trace()))
        assert payload["model"] == ModelID.NANO.value
        assert set(payload["est_credits"]) == {
            ModelID.NANO.value,
            ModelID.MID.value,
            ModelID.PRO.value,
        }


class TestQualityGates:
    def test_high_importance_forces_pro(self):
        decision = select_judge_model(
            claim=SimpleNamespace(importance=0.9, check_worthiness=0.9, harm_potential=4.5),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )
        assert decision.required_capability == "high"
        assert decision.model == ModelID.PRO

    def test_conflicting_evidence_forces_pro(self):
        """Support and refute in the same pack -> quality gate, regardless of cost."""
        decision = select_judge_model(
            claim=_simple_claim(),
            evidence_items=[
                _evidence("apnews.com", stance="support"),
                _evidence("reuters.com", stance="refute"),
            ],
            prompt_chars=4000,
        )
        assert decision.required_capability == "high"
        assert decision.model == ModelID.PRO

    def test_missing_claim_fields_do_not_raise(self):
        decision = select_judge_model(
            claim=SimpleNamespace(),
            evidence_items=[],
            prompt_chars=0,
        )
        assert decision.model in {ModelID.NANO, ModelID.MID, ModelID.PRO}


class TestCostEstimation:
    def test_cost_grows_with_prompt_size(self):
        small = _estimate_credits_for_model(model=ModelID.NANO, prompt_chars=1_000, out_tokens=380)
        large = _estimate_credits_for_model(model=ModelID.NANO, prompt_chars=100_000, out_tokens=380)
        assert large > small

    def test_unknown_model_is_never_preferred(self):
        cost = _estimate_credits_for_model(
            model="not-a-real-model-xyz", prompt_chars=4000, out_tokens=380
        )
        assert cost == UNPRICED_MODEL_CREDITS

    @pytest.mark.parametrize("tier", [ModelID.NANO, ModelID.MID, ModelID.PRO])
    def test_every_configured_tier_has_a_price(self, tier):
        """Guards against a model rename landing without a pricing entry."""
        cost = _estimate_credits_for_model(model=tier, prompt_chars=4000, out_tokens=380)
        assert cost < UNPRICED_MODEL_CREDITS, f"{tier.value} has no entry in default_pricing.json"


class TestDeepSeekFailureBlending:
    def test_fail_probability_raises_mid_expected_cost(self):
        """MID's expected cost must include the PRO fallback it triggers on failure."""
        reliable = select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
            deepseek_fail_prob=0.0,
        )
        flaky = select_judge_model(
            claim=_simple_claim(),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
            deepseek_fail_prob=0.5,
        )

        assert reliable.expected_credits[ModelID.MID] == reliable.est_credits[ModelID.MID]
        assert flaky.expected_credits[ModelID.MID] > reliable.expected_credits[ModelID.MID]
        # NANO and PRO carry no failure blending.
        assert flaky.expected_credits[ModelID.NANO] == flaky.est_credits[ModelID.NANO]


class TestJudgeModelConfig:
    """The general-mode judge dominates run cost (~60% of a typical check),
    so it must stay configurable and must not silently default to the max tier."""

    def test_default_judge_is_high_not_pro(self):
        from spectrue_core.runtime_config import EngineRuntimeConfig

        config = EngineRuntimeConfig.load_from_env()
        assert config.llm.model_judge == ModelID.HIGH
        assert config.llm.model_judge != ModelID.PRO, (
            "defaulting the judge to the max tier is a ~2.5x cost regression"
        )

    def test_judge_model_overridable_via_env(self):
        import os
        from unittest.mock import patch
        from spectrue_core.runtime_config import EngineRuntimeConfig

        with patch.dict(os.environ, {"MODEL_JUDGE": ModelID.PRO.value}, clear=False):
            config = EngineRuntimeConfig.load_from_env()
        assert config.llm.model_judge == ModelID.PRO

    def test_pro_remains_the_escalation_ceiling(self):
        """Cheapening the default must not lower the ceiling for risky claims."""
        decision = select_judge_model(
            claim=SimpleNamespace(importance=0.95, check_worthiness=0.9, harm_potential=4.8),
            evidence_items=_clean_evidence(),
            prompt_chars=4000,
        )
        assert decision.model == ModelID.PRO
        assert decision.fallback_model == ModelID.PRO

    @pytest.mark.parametrize("tier", [ModelID.NANO, ModelID.MID, ModelID.HIGH, ModelID.PRO])
    def test_every_tier_including_high_has_a_price(self, tier):
        cost = _estimate_credits_for_model(model=tier, prompt_chars=4000, out_tokens=380)
        assert cost < UNPRICED_MODEL_CREDITS, f"{tier.value} missing from default_pricing.json"

    def test_high_is_materially_cheaper_than_pro(self):
        high = _estimate_credits_for_model(model=ModelID.HIGH, prompt_chars=14000, out_tokens=1250)
        pro = _estimate_credits_for_model(model=ModelID.PRO, prompt_chars=14000, out_tokens=1250)
        assert high < pro * 0.5, f"expected HIGH well under half of PRO, got {high} vs {pro}"
