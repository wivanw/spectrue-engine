# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from spectrue_core.domain.verification.verdict.model import AnalysisMode
from spectrue_core.domain.claims.model import (
    EvidenceChannel,
    UsePolicy,
    MetadataConfidence,
    ClaimMetadata,
)
# PolicyDecision is in domain.claims.policy
from spectrue_core.domain.claims.policy import ClaimPolicyDecision, PolicyMode
from .types import (
    SearchDepth,
    SearchProfileName,
    StancePassMode,
)
from spectrue_core.domain.verification.verdict.model import resolve_stance_pass_mode

from .policy_models import (
    QualityThresholds,
    LocalePolicy,
    LocaleStrategy,
    SufficiencyThresholds,
    BudgetPolicy,
    SafetyKnobs,
    StopConditions,
    SearchPolicyProfile,
    SearchPolicy,
)

from .ranking import (
    rerank_search_results,
    filter_search_results,
    should_fallback_news_to_general,
    build_context_from_sources,
    prefer_fallback_results,
)

from spectrue_core.domain.verification.verdict.model import AnalysisMode
from spectrue_core.domain.claims.model import (
    EvidenceChannel,
    UsePolicy,
    MetadataConfidence,
    ClaimMetadata,
)
# PolicyDecision is in domain.claims.policy
from spectrue_core.domain.claims.policy import ClaimPolicyDecision, PolicyMode
from .types import (
    SearchDepth,
    SearchProfileName,
    StancePassMode,
)
from spectrue_core.domain.verification.verdict.model import resolve_stance_pass_mode


# Re-export for backward compatibility
__all__ = [
    "SearchDepth",
    "SearchProfileName",
    "StancePassMode",
    "QualityThresholds",
    "LocalePolicy",
    "LocaleStrategy",
    "SufficiencyThresholds",
    "BudgetPolicy",
    "SafetyKnobs",
    "StopConditions",
    "SearchPolicyProfile",
    "SearchPolicy",
    "resolve_profile_name",
    "default_search_policy",
    "decide_claim_policy",
    "resolve_stance_pass_mode",
    "filter_search_results",
    "rerank_search_results",
    "prefer_fallback_results",
    "should_fallback_news_to_general",
    "build_context_from_sources",
]





def resolve_profile_name(mode: AnalysisMode | str | None) -> SearchProfileName:
    """
    Map AnalysisMode to search policy profile name.
    
    Args:
        mode: AnalysisMode enum or string value
        
    Returns:
        SearchProfileName matching available profile
    """
    if mode is None:
        return SearchProfileName.STANDARD
    
    # Handle AnalysisMode enum directly
    if isinstance(mode, AnalysisMode):
        if mode in (AnalysisMode.DEEP, AnalysisMode.DEEP_V2):
            return SearchProfileName.DEEP
        return SearchProfileName.GENERAL
    
    # Handle string (for backward compatibility)
    normalized = str(mode).strip().lower()
    if normalized in (AnalysisMode.DEEP.value, AnalysisMode.DEEP_V2.value):
        return SearchProfileName.DEEP
    return SearchProfileName.GENERAL


def default_search_policy() -> SearchPolicy:
    main_channels = [
        EvidenceChannel.AUTHORITATIVE,
        EvidenceChannel.REPUTABLE_NEWS,
        EvidenceChannel.LOCAL_MEDIA,
    ]
    deep_channels = [
        EvidenceChannel.AUTHORITATIVE,
        EvidenceChannel.REPUTABLE_NEWS,
        EvidenceChannel.LOCAL_MEDIA,
        EvidenceChannel.SOCIAL,
        EvidenceChannel.LOW_RELIABILITY,
    ]
    use_policy_by_channel = {
        EvidenceChannel.AUTHORITATIVE.value: UsePolicy.SUPPORT_OK,
        EvidenceChannel.REPUTABLE_NEWS.value: UsePolicy.SUPPORT_OK,
        EvidenceChannel.LOCAL_MEDIA.value: UsePolicy.SUPPORT_OK,
        EvidenceChannel.SOCIAL.value: UsePolicy.LEAD_ONLY,
        EvidenceChannel.LOW_RELIABILITY.value: UsePolicy.LEAD_ONLY,
    }
    profiles = {
        SearchProfileName.GENERAL.value: SearchPolicyProfile(
            name=SearchProfileName.GENERAL.value,
            search_depth=SearchDepth.BASIC.value,
            max_results=3,
            max_hops=1,
            stance_pass_mode=StancePassMode.SINGLE.value,
            channels_allowed=main_channels,
            use_policy_by_channel=use_policy_by_channel,
            locale_policy=LocalePolicy(primary=None, fallback=None),
            quality_thresholds=QualityThresholds(
                min_relevance_score=0.15,
                min_coverage=0.5,
                min_diversity=0.0,
            ),
        ),
        SearchProfileName.DEEP.value: SearchPolicyProfile(
            name=SearchProfileName.DEEP.value,
            search_depth=SearchDepth.ADVANCED.value,
            max_results=7,
            max_hops=3,
            stance_pass_mode=StancePassMode.TWO_PASS.value,
            channels_allowed=deep_channels,
            use_policy_by_channel=use_policy_by_channel,
            locale_policy=LocalePolicy(primary=None, fallback=None),
            quality_thresholds=QualityThresholds(
                min_relevance_score=0.15,
                min_coverage=0.5,
                min_diversity=0.0,
            ),
        ),
    }
    return SearchPolicy(profiles=profiles)


def decide_claim_policy(metadata: ClaimMetadata | None) -> ClaimPolicyDecision:
    """
    Decide per-claim routing mode (SKIP/CHEAP/FULL) based on metadata signals.
    """
    if metadata is None:
        return ClaimPolicyDecision(mode=PolicyMode.FULL, reason_codes=["metadata_missing"])

    from spectrue_core.domain.claims.policy import should_skip_search
    if should_skip_search(metadata):
        return ClaimPolicyDecision(
            mode=PolicyMode.SKIP,
            reason_codes=["skip_signal"],
        )

    if metadata.metadata_confidence == MetadataConfidence.LOW:
        return ClaimPolicyDecision(
            mode=PolicyMode.CHEAP,
            reason_codes=["low_metadata_confidence"],
        )

    low_threshold = 0.35
    high_threshold = 0.75
    worthiness = float(metadata.check_worthiness or 0.0)

    if worthiness <= low_threshold:
        return ClaimPolicyDecision(
            mode=PolicyMode.CHEAP,
            reason_codes=["low_worthiness"],
        )
    if worthiness >= high_threshold:
        return ClaimPolicyDecision(
            mode=PolicyMode.FULL,
            reason_codes=["high_worthiness"],
        )

    return ClaimPolicyDecision(mode=PolicyMode.FULL, reason_codes=["default_worthiness"])



