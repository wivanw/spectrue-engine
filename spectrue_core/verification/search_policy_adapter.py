from __future__ import annotations

from spectrue_core.constants import (
    DEFAULT_FALLBACK_LOCALES,
    DEFAULT_LOCALE_MAX_FALLBACKS,
    DEFAULT_PRIMARY_LOCALE,
)
from spectrue_core.schema.claim_metadata import SearchLocalePlan, UsePolicy
from spectrue_core.schema.signals import LocaleDecision
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


def evaluate_locale_decision(
    claim: dict,
    profile: SearchPolicyProfile,
    *,
    default_primary_locale: str = DEFAULT_PRIMARY_LOCALE,
    default_fallback_locales: list[str] | None = None,
    max_fallbacks: int = DEFAULT_LOCALE_MAX_FALLBACKS,
) -> LocaleDecision:
    metadata = claim.get("metadata")
    plan = None
    if metadata and getattr(metadata, "search_locale_plan", None):
        plan = metadata.search_locale_plan
    if plan is None:
        plan = SearchLocalePlan(
            primary=default_primary_locale,
            fallback=list(default_fallback_locales or DEFAULT_FALLBACK_LOCALES),
        )

    primary = profile.locale_policy.primary or plan.primary or default_primary_locale

    fallback = (
        profile.locale_policy.fallback
        if profile.locale_policy.fallback is not None
        else plan.fallback
    ) or []
    fallback = [loc for loc in fallback if loc and loc != primary]
    if max_fallbacks >= 0:
        fallback = fallback[:max_fallbacks]

    reason_codes: list[str] = []
    if profile.locale_policy.primary is not None:
        reason_codes.append("primary_policy_override")
    elif metadata:
        reason_codes.append("primary_claim_signal")
    else:
        reason_codes.append("primary_default")

    if profile.locale_policy.fallback is not None:
        reason_codes.append("fallback_policy_override")
    elif metadata:
        reason_codes.append("fallback_claim_signal")
    else:
        reason_codes.append("fallback_default")

    if not fallback:
        reason_codes.append("fallbacks_empty")

    return LocaleDecision(
        primary_locale=primary,
        fallback_locales=fallback,
        used_locales=[primary] if primary else [],
        reason_codes=reason_codes,
        sufficiency_triggered=False,
    )


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


def apply_claim_retrieval_policy(plan: ExecutionPlan, *, claims: list[dict]) -> ExecutionPlan:
    """
    Apply per-claim retrieval policy as a channel whitelist.
    """
    if not claims:
        return plan

    claims_by_id = {str(c.get("id")): c for c in claims if isinstance(c, dict)}

    for claim_id, phases in plan.claim_phases.items():
        claim = claims_by_id.get(str(claim_id))
        metadata = claim.get("metadata") if claim else None
        retrieval_policy = getattr(metadata, "retrieval_policy", None) if metadata else None

        allowed = None
        use_policy_by_channel = None
        if retrieval_policy:
            allowed = getattr(retrieval_policy, "channels_allowed", None)
            use_policy_by_channel = getattr(retrieval_policy, "use_policy_by_channel", None)

        if not allowed:
            continue

        allowed_set = set(allowed)
        for phase in phases:
            phase.channels = [c for c in phase.channels if c in allowed_set]
            if use_policy_by_channel:
                phase.use_policy_by_channel = {
                    c.value: use_policy_by_channel.get(c.value, UsePolicy.LEAD_ONLY)
                    for c in phase.channels
                }

    return plan
