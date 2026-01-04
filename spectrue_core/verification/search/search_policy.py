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

from spectrue_core.schema.claim_metadata import (
    EvidenceChannel,
    UsePolicy,
    MetadataConfidence,
    ClaimMetadata,
)
from spectrue_core.verification.orchestration.execution_plan import ClaimPolicyDecision, PolicyMode


@dataclass(frozen=True)
class QualityThresholds:
    min_relevance_score: float = 0.15
    min_coverage: float = 0.5
    min_diversity: float = 0.0
    rerank_lambda: float = 0.7  # Weight for provider_score in reranking (1-λ for similarity)

    def to_dict(self) -> dict[str, float]:
        return {
            "min_relevance_score": self.min_relevance_score,
            "min_coverage": self.min_coverage,
            "min_diversity": self.min_diversity,
            "rerank_lambda": self.rerank_lambda,
        }


@dataclass(frozen=True)
class LocalePolicy:
    # None preserves per-claim locale plans.
    primary: str | None = None
    fallback: list[str] | None = None

    def to_dict(self) -> dict[str, list[str] | str | None]:
        return {
            "primary": self.primary,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class LocaleStrategy:
    use_primary_first: bool = True
    allow_fallbacks: bool = True
    fallback_on: str = "insufficient"

    def to_dict(self) -> dict[str, object]:
        return {
            "use_primary_first": self.use_primary_first,
            "allow_fallbacks": self.allow_fallbacks,
            "fallback_on": self.fallback_on,
        }


@dataclass(frozen=True)
class SufficiencyThresholds:
    min_support_quotes: int = 1
    min_domain_diversity: int = 1
    min_tier: str = "C"

    def to_dict(self) -> dict[str, object]:
        return {
            "min_support_quotes": self.min_support_quotes,
            "min_domain_diversity": self.min_domain_diversity,
            "min_tier": self.min_tier,
        }


@dataclass(frozen=True)
class BudgetPolicy:
    cost_ceiling_credits: float = 0.0
    per_hop_cost: float = 0.0
    per_fetch_cost: float = 0.0
    per_llm_pass_cost: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "cost_ceiling_credits": self.cost_ceiling_credits,
            "per_hop_cost": self.per_hop_cost,
            "per_fetch_cost": self.per_fetch_cost,
            "per_llm_pass_cost": self.per_llm_pass_cost,
        }


@dataclass(frozen=True)
class SafetyKnobs:
    max_confidence_without_quotes: float = 0.5
    forbid_context_as_support: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "max_confidence_without_quotes": self.max_confidence_without_quotes,
            "forbid_context_as_support": self.forbid_context_as_support,
        }


@dataclass(frozen=True)
class StopConditions:
    stop_on_sufficiency: bool = True
    max_hops: int | None = None

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "stop_on_sufficiency": self.stop_on_sufficiency,
            "max_hops": self.max_hops,
        }


@dataclass(frozen=True)
class SearchPolicyProfile:
    name: str
    search_depth: str = "basic"
    max_results: int = 3
    max_hops: int = 2
    stance_pass_mode: str = "single"
    channels_allowed: list[EvidenceChannel] = field(default_factory=list)
    use_policy_by_channel: dict[str, UsePolicy] = field(default_factory=dict)
    locale_policy: LocalePolicy = field(default_factory=LocalePolicy)
    locale_strategy: LocaleStrategy = field(default_factory=LocaleStrategy)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    stop_conditions: StopConditions = field(default_factory=StopConditions)
    sufficiency_thresholds: SufficiencyThresholds = field(default_factory=SufficiencyThresholds)
    budget_policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    safety_knobs: SafetyKnobs = field(default_factory=SafetyKnobs)
    # Bayesian posterior calibration parameters
    posterior_alpha: float = 1.0  # Weight for LLM signal in posterior
    posterior_beta: float = 1.0   # Weight for evidence signal in posterior

    def __post_init__(self) -> None:
        if self.stop_conditions.max_hops is None:
            object.__setattr__(
                self,
                "stop_conditions",
                StopConditions(
                    stop_on_sufficiency=self.stop_conditions.stop_on_sufficiency,
                    max_hops=self.max_hops,
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "max_hops": self.max_hops,
            "stance_pass_mode": self.stance_pass_mode,
            "channels_allowed": [c.value for c in self.channels_allowed],
            "use_policy_by_channel": {
                k: v.value for k, v in (self.use_policy_by_channel or {}).items()
            },
            "locale_policy": self.locale_policy.to_dict(),
            "locale_strategy": self.locale_strategy.to_dict(),
            "quality_thresholds": self.quality_thresholds.to_dict(),
            "stop_conditions": self.stop_conditions.to_dict(),
            "sufficiency_thresholds": self.sufficiency_thresholds.to_dict(),
            "budget_policy": self.budget_policy.to_dict(),
            "safety_knobs": self.safety_knobs.to_dict(),
            "posterior_alpha": self.posterior_alpha,
            "posterior_beta": self.posterior_beta,
        }


@dataclass(frozen=True)
class SearchPolicy:
    profiles: dict[str, SearchPolicyProfile]

    def get_profile(self, name: str) -> SearchPolicyProfile:
        return self.profiles.get(name, self.profiles["main"])


def resolve_profile_name(raw: str | None) -> str:
    """
    Map search type to policy profile name.
    
    Mapping:
    - "deep", "advanced" → "deep" profile
    - "main", "basic", None → "main" profile
    """
    if not raw:
        return "main"
    normalized = str(raw).strip().lower()
    # Map both "deep" and "advanced" to deep profile
    if normalized in ("deep", "advanced"):
        return "deep"
    if normalized in ("main", "basic"):
        return "main"
    return "main"


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
        "main": SearchPolicyProfile(
            name="main",
            search_depth="basic",
            max_results=3,
            max_hops=1,
            stance_pass_mode="single",
            channels_allowed=main_channels,
            use_policy_by_channel=use_policy_by_channel,
            locale_policy=LocalePolicy(primary=None, fallback=None),
            quality_thresholds=QualityThresholds(
                min_relevance_score=0.15,
                min_coverage=0.5,
                min_diversity=0.0,
            ),
        ),
        "deep": SearchPolicyProfile(
            name="deep",
            search_depth="advanced",
            max_results=7,
            max_hops=3,
            stance_pass_mode="two_pass",
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

    if metadata.should_skip_search:
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


def resolve_stance_pass_mode(profile_name: str) -> str:
    normalized = (profile_name or "main").strip().lower()
    return "two_pass" if normalized == "deep" else "single"


def rerank_search_results(
    results: list[dict],
    *,
    rerank_lambda: float = 0.7,
    top_k: int | None = None,
    skip_extensions: tuple[str, ...] = (".txt", ".xml", ".zip"),
) -> list[dict]:
    """
    Rerank search results using combined score instead of hard filtering.
    
    Formula: score = λ · provider_score + (1-λ) · relevance_score
    
    This replaces hard threshold filtering which could discard valid results.
    Results are sorted by combined score and optionally limited to top_k.
    
    Args:
        results: List of search result dicts with 'score' and 'relevance_score'
        rerank_lambda: Weight for provider score (default 0.7)
        top_k: If set, return only top K results
        skip_extensions: File extensions to always skip
        
    Returns:
        Reranked list of results (no hard filtering, only sorting)
    """
    from spectrue_core.utils.trace import Trace

    scored: list[tuple[float, dict]] = []

    for r in (results or []):
        # Skip problematic file types
        url_str = r.get("link", "") or r.get("url", "")
        if isinstance(url_str, str) and url_str.lower().endswith(skip_extensions):
            continue

        # Get scores with defaults
        provider_score = r.get("score")
        if not isinstance(provider_score, (int, float)):
            provider_score = 0.5
        provider_score = max(0.0, min(1.0, float(provider_score)))

        relevance_score = r.get("relevance_score")
        if not isinstance(relevance_score, (int, float)):
            relevance_score = provider_score  # Fallback to provider score
        relevance_score = max(0.0, min(1.0, float(relevance_score)))

        # Combined score: λ · provider + (1-λ) · relevance
        combined = rerank_lambda * provider_score + (1 - rerank_lambda) * relevance_score

        # Store for sorting
        r["_rerank_score"] = combined
        scored.append((combined, r))

    # Sort by combined score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Apply top_k limit if set
    if top_k is not None and top_k > 0:
        scored = scored[:top_k]

    out = [r for _, r in scored]

    Trace.event(
        "search.rerank",
        {
            "input_count": len(results or []),
            "output_count": len(out),
            "rerank_lambda": rerank_lambda,
            "top_k": top_k,
            "top_scores": [r.get("_rerank_score") for r in out[:3]] if out else [],
        },
    )

    return out


def filter_search_results(
    results: list[dict],
    *,
    min_relevance_score: float = 0.15,
    skip_extensions: tuple[str, ...] = (".txt", ".xml", ".zip"),
) -> list[dict]:
    """
    Legacy filter function - kept for backward compatibility.
    
    For new code, prefer rerank_search_results() which doesn't discard results.
    """
    out: list[dict] = []
    for r in (results or []):
        score = r.get("relevance_score")
        if isinstance(score, (int, float)) and float(score) < float(min_relevance_score):
            continue

        url_str = r.get("link", "") or r.get("url", "")
        if isinstance(url_str, str) and url_str.lower().endswith(skip_extensions):
            continue

        out.append(r)
    return out


def should_fallback_news_to_general(topic: str, filtered: list[dict]) -> tuple[bool, str, float]:
    if topic != "news":
        return False, "", 0.0
    valid_count = len(filtered or [])
    max_score = max([float(r.get("score", 0) or 0.0) for r in (filtered or [])]) if filtered else 0.0
    if valid_count < 2:
        return True, f"few_results ({valid_count})", max_score
    if max_score < 0.2:
        return True, f"low_relevance ({max_score:.2f})", max_score
    return False, "", max_score


def build_context_from_sources(sources: Iterable[dict]) -> str:
    def format_source(obj: dict) -> str:
        return f"Source: {obj.get('title')}\nURL: {obj.get('link')}\nContent: {obj.get('snippet')}\n---"

    return "\n".join([format_source(obj) for obj in (sources or [])])


def prefer_fallback_results(
    *,
    original_filtered: list[dict],
    original_max_score: float,
    fallback_filtered: list[dict],
) -> bool:
    fb_count = len(fallback_filtered or [])
    fb_max_score = max([float(r.get("score", 0) or 0.0) for r in (fallback_filtered or [])]) if fallback_filtered else 0.0

    return fb_count > 0 and (len(original_filtered or []) == 0 or fb_max_score > original_max_score)
