from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.domain.claims.model import EvidenceChannel, UsePolicy

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
    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0

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
    """Container of named search profiles (e.g. general, deep)."""

    profiles: dict[str, SearchPolicyProfile] = field(default_factory=dict)

    def get_profile(self, profile_name: str | Any) -> SearchPolicyProfile | None:
        """Return profile by name; accepts enum (uses .value) or str. Fallback to 'general' if missing."""
        key = getattr(profile_name, "value", profile_name) if not isinstance(profile_name, str) else profile_name
        key = (key or "").strip().lower() or "general"
        return self.profiles.get(key) or self.profiles.get("general")
