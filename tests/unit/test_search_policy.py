# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.



from spectrue_core.schema.claim_metadata import ClaimMetadata, EvidenceChannel, RetrievalPolicy
from spectrue_core.verification.orchestration.execution_plan import BudgetClass, ExecutionPlan, Phase
from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
    SearchPolicyProfile,
)
from spectrue_core.verification.search.search_policy_adapter import (
    apply_claim_retrieval_policy,
    apply_search_policy_to_plan,
    budget_class_for_profile,
)


def _make_claim(claim_id: str) -> dict:
    return {
        "id": claim_id,
        "text": "Sample claim",
        "normalized_text": "Sample claim",
        "metadata": ClaimMetadata(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# resolve_profile_name() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveProfileName:
    """Tests for resolve_profile_name() function."""

    def test_none_returns_general(self):
        """None input should return 'general' profile."""
        assert resolve_profile_name(None) == "general"

    def test_empty_string_returns_general(self):
        """Empty string should return 'general' profile."""
        assert resolve_profile_name("") == "general"

    def test_general_returns_general(self):
        """'general' should return 'general' profile."""
        assert resolve_profile_name("general") == "general"

    def test_deep_returns_deep(self):
        """'deep' should return 'deep' profile."""
        assert resolve_profile_name("deep") == "deep"

    def test_deep_v2_returns_deep(self):
        """'deep_v2' should return 'deep' profile."""
        assert resolve_profile_name("deep_v2") == "deep"

    def test_case_insensitive(self):
        """Profile name resolution should be case-insensitive."""
        assert resolve_profile_name("DEEP") == "deep"
        assert resolve_profile_name("DEEP_V2") == "deep"
        assert resolve_profile_name("GENERAL") == "general"

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed from input."""
        assert resolve_profile_name("  deep  ") == "deep"
        assert resolve_profile_name("\tdeep_v2\n") == "deep"

    def test_unknown_returns_general(self):
        """Unknown profile names should default to 'general'."""
        assert resolve_profile_name("unknown") == "general"
        assert resolve_profile_name("custom") == "general"

    def test_analysis_mode_enum_general(self):
        """AnalysisMode.GENERAL enum should return 'general' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.GENERAL) == "general"

    def test_analysis_mode_enum_deep(self):
        """AnalysisMode.DEEP enum should return 'deep' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.DEEP) == "deep"

    def test_analysis_mode_enum_deep_v2(self):
        """AnalysisMode.DEEP_V2 enum should return 'deep' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.DEEP_V2) == "deep"


# ─────────────────────────────────────────────────────────────────────────────
# budget_class_for_profile() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetClassForProfile:
    """Tests for budget_class_for_profile() function."""

    def test_general_profile_returns_minimal(self):
        """General profile (max_hops=1) should return MINIMAL budget class."""
        policy = default_search_policy()
        general_profile = policy.get_profile("general")
        assert general_profile.max_hops == 1
        assert budget_class_for_profile(general_profile) == BudgetClass.MINIMAL

    def test_deep_profile_returns_deep(self):
        """Deep profile (max_hops=3) should return DEEP budget class (M120 fix)."""
        policy = default_search_policy()
        deep_profile = policy.get_profile("deep")
        assert deep_profile.max_hops == 3
        assert budget_class_for_profile(deep_profile) == BudgetClass.DEEP

    def test_max_hops_none_returns_standard(self):
        """Profile with max_hops=None should return STANDARD budget class."""
        profile = SearchPolicyProfile(name="test", max_hops=None)
        # Note: max_hops=None triggers __post_init__ to use class default (2)
        # But we need to check the raw logic, so we mock
        # Actually, __post_init__ sets stop_conditions.max_hops to self.max_hops
        # Let's verify the function directly handles None
        
        # Create a profile where we can control max_hops more directly
        # The function checks profile.max_hops, not stop_conditions
        assert budget_class_for_profile(profile) == BudgetClass.STANDARD

    def test_max_hops_1_returns_minimal(self):
        """Profile with max_hops=1 should return MINIMAL."""
        profile = SearchPolicyProfile(name="test", max_hops=1)
        assert budget_class_for_profile(profile) == BudgetClass.MINIMAL

    def test_max_hops_2_returns_standard(self):
        """Profile with max_hops=2 should return STANDARD."""
        profile = SearchPolicyProfile(name="test", max_hops=2)
        assert budget_class_for_profile(profile) == BudgetClass.STANDARD

    def test_max_hops_3_returns_deep(self):
        """Profile with max_hops=3 should return DEEP (M120 threshold fix)."""
        profile = SearchPolicyProfile(name="test", max_hops=3)
        assert budget_class_for_profile(profile) == BudgetClass.DEEP

    def test_max_hops_4_returns_deep(self):
        """Profile with max_hops=4 should return DEEP."""
        profile = SearchPolicyProfile(name="test", max_hops=4)
        assert budget_class_for_profile(profile) == BudgetClass.DEEP

    def test_max_hops_10_returns_deep(self):
        """Profile with high max_hops should return DEEP."""
        profile = SearchPolicyProfile(name="test", max_hops=10)
        assert budget_class_for_profile(profile) == BudgetClass.DEEP


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Profile → Budget Class → Plan
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchPolicyIntegration:
    """Integration tests for search policy → execution plan."""

    def test_search_policy_profiles_diverge(self):
        """General and deep profiles should produce different execution plans."""
        orchestrator = ClaimOrchestrator()
        claims = [_make_claim("c1")]

        policy = default_search_policy()
        general_profile = policy.get_profile("general")
        deep_profile = policy.get_profile("deep")

        general_plan = orchestrator.build_plan(
            claims,
            budget_class=budget_class_for_profile(general_profile),
        )
        general_plan = apply_search_policy_to_plan(general_plan, profile=general_profile)

        deep_plan = orchestrator.build_plan(
            claims,
            budget_class=budget_class_for_profile(deep_profile),
        )
        deep_plan = apply_search_policy_to_plan(deep_plan, profile=deep_profile)

        general_phases = general_plan.get_phases("c1")
        deep_phases = deep_plan.get_phases("c1")

        assert len(general_phases) < len(deep_phases)

        general_depths = {p.search_depth for p in general_phases}
        deep_depths = {p.search_depth for p in deep_phases}
        assert general_depths != deep_depths

        general_max_results = max(p.max_results for p in general_phases)
        deep_max_results = max(p.max_results for p in deep_phases)
        assert general_max_results < deep_max_results

    def test_deep_v2_analysis_mode_gets_deep_budget(self):
        """analysis_mode='deep_v2' should resolve to deep profile with DEEP budget."""
        profile_name = resolve_profile_name("deep_v2")
        assert profile_name == "deep"

        policy = default_search_policy()
        profile = policy.get_profile(profile_name)
        budget_class = budget_class_for_profile(profile)

        assert budget_class == BudgetClass.DEEP
        assert profile.max_hops == 3
        assert profile.max_results == 7

    def test_general_analysis_mode_gets_minimal_budget(self):
        """analysis_mode='general' should resolve to general profile with MINIMAL budget."""
        profile_name = resolve_profile_name("general")
        assert profile_name == "general"

        policy = default_search_policy()
        profile = policy.get_profile(profile_name)
        budget_class = budget_class_for_profile(profile)

        assert budget_class == BudgetClass.MINIMAL
        assert profile.max_hops == 1
        assert profile.max_results == 3


# ─────────────────────────────────────────────────────────────────────────────
# apply_claim_retrieval_policy() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClaimRetrievalPolicy:
    """Tests for per-claim retrieval policy mapping and handling."""

    def _make_plan(self) -> ExecutionPlan:
        phase = Phase(
            phase_id="A",
            locale="en",
            channels=[
                EvidenceChannel.AUTHORITATIVE,
                EvidenceChannel.REPUTABLE_NEWS,
                EvidenceChannel.SOCIAL,
            ],
            search_depth="basic",
            max_results=3,
        )
        return ExecutionPlan(
            claim_phases={"c1": [phase]},
            budget_class=BudgetClass.STANDARD,
        )

    def test_maps_string_channels_case_insensitive(self):
        plan = self._make_plan()
        claim = {
            "id": "c1",
            "metadata": ClaimMetadata(
                retrieval_policy=RetrievalPolicy(
                    channels_allowed=["Authoritative", "REPUTABLE_NEWS"]
                )
            ),
        }

        updated = apply_claim_retrieval_policy(plan, claims=[claim])
        channels = updated.get_phases("c1")[0].channels

        assert channels == [
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ]

    def test_unknown_tokens_do_not_clear_channels(self):
        plan = self._make_plan()
        original = list(plan.get_phases("c1")[0].channels)
        claim = {
            "id": "c1",
            "metadata": ClaimMetadata(
                retrieval_policy=RetrievalPolicy(
                    channels_allowed=["unknown_token", "also_bad"]
                )
            ),
        }

        updated = apply_claim_retrieval_policy(plan, claims=[claim])
        channels = updated.get_phases("c1")[0].channels

        assert channels == original


# Keep original test for backward compatibility
def test_search_policy_profiles_diverge():
    """Original test - kept for backward compatibility."""
    orchestrator = ClaimOrchestrator()
    claims = [_make_claim("c1")]

    policy = default_search_policy()
    general_profile = policy.get_profile("general")
    deep_profile = policy.get_profile("deep")

    general_plan = orchestrator.build_plan(
        claims,
        budget_class=budget_class_for_profile(general_profile),
    )
    general_plan = apply_search_policy_to_plan(general_plan, profile=general_profile)

    deep_plan = orchestrator.build_plan(
        claims,
        budget_class=budget_class_for_profile(deep_profile),
    )
    deep_plan = apply_search_policy_to_plan(deep_plan, profile=deep_profile)

    general_phases = general_plan.get_phases("c1")
    deep_phases = deep_plan.get_phases("c1")

    assert len(general_phases) < len(deep_phases)

    general_depths = {p.search_depth for p in general_phases}
    deep_depths = {p.search_depth for p in deep_phases}
    assert general_depths != deep_depths

    general_max_results = max(p.max_results for p in general_phases)
    deep_max_results = max(p.max_results for p in deep_phases)
    assert general_max_results < deep_max_results
