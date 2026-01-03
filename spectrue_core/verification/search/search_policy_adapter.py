# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from typing import TYPE_CHECKING

from spectrue_core.constants import (
    DEFAULT_FALLBACK_LOCALES,
    DEFAULT_LOCALE_MAX_FALLBACKS,
    DEFAULT_PRIMARY_LOCALE,
)
from spectrue_core.schema.claim_metadata import EvidenceChannel, SearchLocalePlan, UsePolicy
from spectrue_core.schema.signals import LocaleDecision
from spectrue_core.verification.orchestration.execution_plan import BudgetClass, ExecutionPlan, Phase
from spectrue_core.verification.search.search_policy import LocalePolicy, SearchPolicyProfile

if TYPE_CHECKING:
    from spectrue_core.pipeline_builder.spec import PipelineProfile


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
    """
    Derive budget class from profile's max_hops setting.
    
    Thresholds:
    - max_hops <= 1: MINIMAL (fast, limited search)
    - max_hops >= 3: DEEP (thorough search)
    - otherwise: STANDARD (balanced)
    """
    max_hops = profile.max_hops
    if max_hops is None:
        return BudgetClass.STANDARD
    if max_hops <= 1:
        return BudgetClass.MINIMAL
    if max_hops >= 3:
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


# ─────────────────────────────────────────────────────────────────────────────
# M113: Pipeline Profile Adapter
# ─────────────────────────────────────────────────────────────────────────────


def apply_pipeline_profile_to_plan(
    plan: ExecutionPlan,
    *,
    pipeline_profile: "PipelineProfile",
) -> ExecutionPlan:
    """
    Apply pipeline profile settings to an execution plan.

    This function adapts settings from a M113 PipelineProfile to an ExecutionPlan,
    ensuring that profile constraints are respected.

    Args:
        plan: ExecutionPlan to modify
        pipeline_profile: PipelineProfile with settings to apply

    Returns:
        Modified ExecutionPlan with profile settings applied
    """
    from spectrue_core.pipeline_builder.spec import PipelineProfile

    if not isinstance(pipeline_profile, PipelineProfile):
        return plan

    # Apply search depth cap
    search_depth = pipeline_profile.search.depth
    max_results = pipeline_profile.search.max_results

    # Get allowed channels from profile
    allowed_channels_str = set(pipeline_profile.channels.allowed)
    blocked_channels_str = set(pipeline_profile.channels.blocked)

    # Convert to EvidenceChannel
    channel_map = {
        "authoritative": EvidenceChannel.AUTHORITATIVE,
        "reputable_news": EvidenceChannel.REPUTABLE_NEWS,
        "local_media": EvidenceChannel.LOCAL_MEDIA,
        "social": EvidenceChannel.SOCIAL,
        "low_reliability": EvidenceChannel.LOW_RELIABILITY,
    }
    allowed_channels = {channel_map.get(c) for c in allowed_channels_str if c in channel_map}
    blocked_channels = {channel_map.get(c) for c in blocked_channels_str if c in channel_map}
    allowed_channels.discard(None)
    blocked_channels.discard(None)

    # Apply to each phase
    for claim_id, phases in plan.claim_phases.items():
        for phase in phases:
            # Apply search depth
            phase.search_depth = search_depth
            phase.max_results = min(phase.max_results, max_results)
            phase.is_expensive = phase.search_depth == "advanced" or phase.max_results >= 5

            # Filter channels
            if allowed_channels:
                phase.channels = [c for c in phase.channels if c in allowed_channels]
            if blocked_channels:
                phase.channels = [c for c in phase.channels if c not in blocked_channels]

    # Apply max depth from profile
    max_depth = pipeline_profile.phases.max_depth
    if max_depth > 0:
        for claim_id in plan.claim_phases:
            plan.claim_phases[claim_id] = plan.claim_phases[claim_id][:max_depth]

    return plan

