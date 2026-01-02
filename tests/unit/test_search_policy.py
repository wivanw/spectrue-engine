# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.schema.claim_metadata import ClaimMetadata
from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
from spectrue_core.verification.search.search_policy import default_search_policy
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


def test_search_policy_profiles_diverge():
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
