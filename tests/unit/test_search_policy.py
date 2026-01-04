# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest

from spectrue_core.schema.claim_metadata import ClaimMetadata
from spectrue_core.verification.orchestration.execution_plan import BudgetClass
from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
    SearchPolicyProfile,
)
from spectrue_core.verification.search.search_policy_adapter import (
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

    def test_none_returns_main(self):
        """None input should return 'main' profile."""
        assert resolve_profile_name(None) == "main"

    def test_empty_string_returns_main(self):
        """Empty string should return 'main' profile."""
        assert resolve_profile_name("") == "main"

    def test_main_returns_main(self):
        """'main' should return 'main' profile."""
        assert resolve_profile_name("main") == "main"

    def test_basic_returns_main(self):
        """'basic' should return 'main' profile."""
        assert resolve_profile_name("basic") == "main"

    def test_deep_returns_deep(self):
        """'deep' should return 'deep' profile."""
        assert resolve_profile_name("deep") == "deep"

    def test_advanced_returns_deep(self):
        """'advanced' should return 'deep' profile (M120 fix)."""
        assert resolve_profile_name("advanced") == "deep"

    def test_case_insensitive(self):
        """Profile name resolution should be case-insensitive."""
        assert resolve_profile_name("DEEP") == "deep"
        assert resolve_profile_name("Advanced") == "deep"
        assert resolve_profile_name("MAIN") == "main"
        assert resolve_profile_name("Basic") == "main"

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed from input."""
        assert resolve_profile_name("  deep  ") == "deep"
        assert resolve_profile_name("\tadvanced\n") == "deep"

    def test_unknown_returns_main(self):
        """Unknown profile names should default to 'main'."""
        assert resolve_profile_name("unknown") == "main"
        assert resolve_profile_name("custom") == "main"


# ─────────────────────────────────────────────────────────────────────────────
# budget_class_for_profile() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetClassForProfile:
    """Tests for budget_class_for_profile() function."""

    def test_main_profile_returns_minimal(self):
        """Main profile (max_hops=1) should return MINIMAL budget class."""
        policy = default_search_policy()
        main_profile = policy.get_profile("main")
        assert main_profile.max_hops == 1
        assert budget_class_for_profile(main_profile) == BudgetClass.MINIMAL

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
        """Main and deep profiles should produce different execution plans."""
        orchestrator = ClaimOrchestrator()
        claims = [_make_claim("c1")]

        policy = default_search_policy()
        main_profile = policy.get_profile("main")
        deep_profile = policy.get_profile("deep")

        main_plan = orchestrator.build_plan(
            claims,
            budget_class=budget_class_for_profile(main_profile),
        )
        main_plan = apply_search_policy_to_plan(main_plan, profile=main_profile)

        deep_plan = orchestrator.build_plan(
            claims,
            budget_class=budget_class_for_profile(deep_profile),
        )
        deep_plan = apply_search_policy_to_plan(deep_plan, profile=deep_profile)

        main_phases = main_plan.get_phases("c1")
        deep_phases = deep_plan.get_phases("c1")

        assert len(main_phases) < len(deep_phases)

        main_depths = {p.search_depth for p in main_phases}
        deep_depths = {p.search_depth for p in deep_phases}
        assert main_depths != deep_depths

        main_max_results = max(p.max_results for p in main_phases)
        deep_max_results = max(p.max_results for p in deep_phases)
        assert main_max_results < deep_max_results

    def test_advanced_search_type_gets_deep_budget(self):
        """search_type='advanced' should resolve to deep profile with DEEP budget."""
        profile_name = resolve_profile_name("advanced")
        assert profile_name == "deep"

        policy = default_search_policy()
        profile = policy.get_profile(profile_name)
        budget_class = budget_class_for_profile(profile)

        assert budget_class == BudgetClass.DEEP
        assert profile.max_hops == 3
        assert profile.max_results == 7

    def test_basic_search_type_gets_minimal_budget(self):
        """search_type='basic' should resolve to main profile with MINIMAL budget."""
        profile_name = resolve_profile_name("basic")
        assert profile_name == "main"

        policy = default_search_policy()
        profile = policy.get_profile(profile_name)
        budget_class = budget_class_for_profile(profile)

        assert budget_class == BudgetClass.MINIMAL
        assert profile.max_hops == 1
        assert profile.max_results == 3


# Keep original test for backward compatibility
def test_search_policy_profiles_diverge():
    """Original test - kept for backward compatibility."""
    orchestrator = ClaimOrchestrator()
    claims = [_make_claim("c1")]

    policy = default_search_policy()
    main_profile = policy.get_profile("main")
    deep_profile = policy.get_profile("deep")

    main_plan = orchestrator.build_plan(
        claims,
        budget_class=budget_class_for_profile(main_profile),
    )
    main_plan = apply_search_policy_to_plan(main_plan, profile=main_profile)

    deep_plan = orchestrator.build_plan(
        claims,
        budget_class=budget_class_for_profile(deep_profile),
    )
    deep_plan = apply_search_policy_to_plan(deep_plan, profile=deep_profile)

    main_phases = main_plan.get_phases("c1")
    deep_phases = deep_plan.get_phases("c1")

    assert len(main_phases) < len(deep_phases)

    main_depths = {p.search_depth for p in main_phases}
    deep_depths = {p.search_depth for p in deep_phases}
    assert main_depths != deep_depths

    main_max_results = max(p.max_results for p in main_phases)
    deep_max_results = max(p.max_results for p in deep_phases)
    assert main_max_results < deep_max_results
