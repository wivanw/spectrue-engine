from __future__ import annotations

from spectrue_core.schema.claim_metadata import UsePolicy
from spectrue_core.verification.execution_plan import BudgetClass, ExecutionPlan, Phase
from spectrue_core.verification.search_policy import LocalePolicy, SearchPolicyProfile


def _resolve_locale(phase_id: str, locale_policy: LocalePolicy, current: str) -> str:
    if locale_policy.primary is None and not locale_policy.fallback:
        return current

    if phase_id in ("C",):
        if locale_policy.fallback:
            return locale_policy.fallback[0]
        return current

    if locale_policy.primary is not None:
        return locale_policy.primary

    return current


def _cap_search_depth(phase: Phase, search_depth: str) -> None:
    phase.search_depth = search_depth
    phase.is_expensive = phase.search_depth == "advanced" or phase.max_results >= 5


def _cap_max_results(phase: Phase, max_results: int) -> None:
    if max_results <= 0:
        return
    phase.max_results = min(phase.max_results, max_results)
    phase.is_expensive = phase.search_depth == "advanced" or phase.max_results >= 5


def _apply_policy_channels(phase: Phase, profile: SearchPolicyProfile) -> None:
    allowed = set(profile.channels_allowed or [])
    if allowed:
        phase.channels = [c for c in phase.channels if c in allowed]

    phase.use_policy_by_channel = {
        c.value: profile.use_policy_by_channel.get(c.value, UsePolicy.LEAD_ONLY)
        for c in phase.channels
    }


def _apply_policy_to_phase(phase: Phase, profile: SearchPolicyProfile) -> Phase | None:
    phase.locale = _resolve_locale(phase.phase_id, profile.locale_policy, phase.locale)
    _cap_search_depth(phase, profile.search_depth)
    _cap_max_results(phase, profile.max_results)
    _apply_policy_channels(phase, profile)

    if not phase.channels:
        return None

    return phase


def budget_class_for_profile(profile: SearchPolicyProfile) -> BudgetClass:
    max_hops = profile.max_hops
    if max_hops is None:
        return BudgetClass.STANDARD
    if max_hops <= 1:
        return BudgetClass.MINIMAL
    if max_hops >= 4:
        return BudgetClass.DEEP
    return BudgetClass.STANDARD


def apply_search_policy_to_plan(
    plan: ExecutionPlan,
    *,
    profile: SearchPolicyProfile,
) -> ExecutionPlan:
    capped: dict[str, list[Phase]] = {}
    max_hops = profile.max_hops

    for claim_id, phases in plan.claim_phases.items():
        next_phases: list[Phase] = []
        for phase in phases:
            updated = _apply_policy_to_phase(phase, profile)
            if updated is not None:
                next_phases.append(updated)

        if max_hops is not None and max_hops > 0:
            next_phases = next_phases[:max_hops]

        capped[claim_id] = next_phases

    plan.claim_phases = capped
    plan.budget_class = budget_class_for_profile(profile)

    return plan
