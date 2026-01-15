# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Claim-Centric Orchestration Metadata Types

This module defines metadata types for claim-level verification routing:
- ClaimRole: Role of claim in document structure
- VerificationTarget: What aspect to verify (reality/attribution/existence/none)
- SearchLocalePlan: Language strategy for search
- RetrievalPolicy: Allowed evidence channels and usage modes
- ClaimMetadata: Complete metadata for orchestration decisions

Design Principles:
1. Metadata-driven routing (no genre heuristics like "if horoscope...")
2. LLM fills metadata at extraction time
3. Orchestrator uses metadata to build ExecutionPlan
4. Fail-open: low confidence → don't skip, do minimal search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ClaimRole(str, Enum):
    """
    Role of a claim in the document structure.
    
    Affects RGBA aggregation weight and explanation inclusion.
    - CORE/SUPPORT/ATTRIBUTION: Full weight in scoring
    - CONTEXT/META: Weight=0, explain-only
    """
    CORE = "core"
    """Central claim of the article. Highest verification priority."""

    SUPPORT = "support"
    """Supporting evidence for a core claim."""

    CONTEXT = "context"
    """Background information. Not a fact to verify."""

    META = "meta"
    """Information about the article/source itself."""

    ATTRIBUTION = "attribution"
    """Quote attribution: who said what."""

    AGGREGATED = "aggregated"
    """Summary derived from multiple sources."""

    SUBCLAIM = "subclaim"
    """Subordinate detail within a larger claim."""

    # Structured document roles
    THESIS = "thesis"
    """Main thesis or conclusion of the article."""

    BACKGROUND = "background"
    """Background context for the topic."""

    EXAMPLE = "example"
    """Illustrative example supporting another claim."""

    HEDGE = "hedge"
    """Hedged/qualified statement ("may", "might", "possibly")."""

    COUNTERCLAIM = "counterclaim"
    """Counterpoint or opposing claim."""


class VerificationTarget(str, Enum):
    """
    What aspect of the claim to verify.
    
    Determines search strategy and verdict semantics.
    - REALITY: Is it factually true? (standard verification)
    - ATTRIBUTION: Did X really say Y?
    - EXISTENCE: Does the source/document exist?
    - NONE: Not verifiable (predictions, opinions)
    """
    REALITY = "reality"
    """Verify factual accuracy against reality. Standard path."""

    ATTRIBUTION = "attribution"
    """Verify that person X said/did thing Y."""

    EXISTENCE = "existence"
    """Verify that a source/document/entity exists."""

    NONE = "none"
    """Not verifiable. Predictions, opinions, horoscopes, subjective."""


class EvidenceChannel(str, Enum):
    """
    Evidence source tier/channel.
    
    Used for retrieval policy and sufficiency checks.
    Higher tiers = more authoritative = higher weight.
    """
    AUTHORITATIVE = "authoritative"
    """Official sources: .gov, .edu, WHO, CDC, peer-reviewed journals."""

    REPUTABLE_NEWS = "reputable_news"
    """Major news outlets: Reuters, AP, BBC, NYT, etc."""

    LOCAL_MEDIA = "local_media"
    """Regional/local news sources. Good for local events."""

    SOCIAL = "social"
    """Social media: Twitter, Reddit, Facebook. Lead-only by default."""

    LOW_RELIABILITY = "low_reliability_web"
    """Blogs, forums, unknown sites. Lead-only by default, capped weight."""


class UsePolicy(str, Enum):
    """
    How evidence from a channel can be used.
    
    - SUPPORT_OK: Can directly support/refute claims
    - LEAD_ONLY: Can only be used as leads for further search
    """
    SUPPORT_OK = "support_ok"
    """Channel can provide evidence that supports/refutes claims."""

    LEAD_ONLY = "lead_only"
    """Channel can only provide leads, not definitive evidence."""


class MetadataConfidence(str, Enum):
    """
    Confidence in the extracted metadata.
    
    Used for fail-open decisions:
    - LOW: Trigger fail-open (don't skip, do Phase A-light)
    - MEDIUM: Normal processing
    - HIGH: Trust metadata fully
    """
    LOW = "low"
    """Low confidence. Trigger fail-open: always do Phase A-light."""

    MEDIUM = "medium"
    """Medium confidence. Normal processing."""

    HIGH = "high"
    """High confidence. Trust metadata for full routing."""


def _normalize_channel_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchLocalePlan:
    """
    Language/locale strategy for search queries.
    
    LLM decides based on claim content:
    - Scientific claims → primary="en"
    - Local news → primary=[article language]
    - International topics → include "en" in fallback
    
    NOTE: UI locale ≠ search locale. UI locale is for explanations only.
    """
    primary: str = "en"
    """Primary search locale. Used in Phase A/B."""

    fallback: list[str] = field(default_factory=lambda: ["en"])
    """Fallback locales. Used in Phase C/D if primary yields insufficient evidence."""

    def __post_init__(self) -> None:
        # Ensure fallback is a list
        if not isinstance(self.fallback, list):
            self.fallback = [self.fallback] if self.fallback else ["en"]


@dataclass
class RetrievalPolicy:
    """
    Policy for evidence retrieval: which channels are allowed and how.
    
    LLM sets this based on claim type and harm potential:
    - High harm (medical/financial) → authoritative only
    - Attribution claims → include original source channel
    - Low-risk evergreen → allow wider channels
    
    Default: authoritative + reputable_news as SUPPORT_OK,
             social + low_reliability_web as LEAD_ONLY.
    """
    channels_allowed: list[EvidenceChannel] = field(
        default_factory=lambda: [
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
            EvidenceChannel.LOCAL_MEDIA,
        ]
    )
    """Which channels can be searched."""

    use_policy: dict[str, UsePolicy] = field(
        default_factory=lambda: {
            EvidenceChannel.AUTHORITATIVE.value: UsePolicy.SUPPORT_OK,
            EvidenceChannel.REPUTABLE_NEWS.value: UsePolicy.SUPPORT_OK,
            EvidenceChannel.LOCAL_MEDIA.value: UsePolicy.SUPPORT_OK,
            EvidenceChannel.SOCIAL.value: UsePolicy.LEAD_ONLY,
            EvidenceChannel.LOW_RELIABILITY.value: UsePolicy.LEAD_ONLY,
        }
    )
    """Per-channel usage policy."""

    @property
    def use_policy_by_channel(self) -> dict[str, UsePolicy]:
        """Canonical alias for spec terminology."""
        return self.use_policy

    def get_use_policy(self, channel: EvidenceChannel) -> UsePolicy:
        """Get usage policy for a channel. Defaults to LEAD_ONLY."""
        return UsePolicy(
            self.use_policy.get(channel.value, UsePolicy.LEAD_ONLY.value)
        )

    def can_support(self, channel: EvidenceChannel) -> bool:
        """Check if channel can provide supporting evidence."""
        return self.get_use_policy(channel) == UsePolicy.SUPPORT_OK


@dataclass
class ClaimMetadata:
    """
    Complete metadata for claim-centric orchestration.
    
    This is the OUTPUT of claim extraction (filled by LLM).
    This is the INPUT to the orchestrator (builds ExecutionPlan).
    
    Example (factual news claim):
        ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.9,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=["uk"]),
            retrieval_policy=RetrievalPolicy(...),
            metadata_confidence=MetadataConfidence.HIGH
        )
    
    Example (horoscope prediction):
        ClaimMetadata(
            verification_target=VerificationTarget.NONE,
            claim_role=ClaimRole.CONTEXT,
            check_worthiness=0.1,
            search_locale_plan=SearchLocalePlan(primary="en"),
            retrieval_policy=RetrievalPolicy(channels_allowed=[]),
            metadata_confidence=MetadataConfidence.HIGH
        )
    """
    verification_target: VerificationTarget = VerificationTarget.REALITY
    """What to verify: reality/attribution/existence/none."""

    claim_role: ClaimRole = ClaimRole.CORE
    """Role in document: core/support/context/meta/attribution/aggregated/subclaim."""

    check_worthiness: float = 0.5
    """Priority for verification budget. 0=skip, 1=must verify."""

    search_locale_plan: SearchLocalePlan = field(default_factory=SearchLocalePlan)
    """Language strategy for search queries."""

    time_signals: list[dict[str, Any]] = field(default_factory=list)
    """LLM-extracted temporal anchors (may be empty)."""

    locale_signals: list[dict[str, Any]] = field(default_factory=list)
    """LLM-extracted locale anchors (may be empty)."""

    time_sensitive: bool = False
    """True when claim includes explicit temporal anchors or recency signals."""

    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    """Allowed channels and usage modes."""

    metadata_confidence: MetadataConfidence = MetadataConfidence.MEDIUM
    """Confidence in this metadata. LOW triggers fail-open."""

    is_key_claim: bool = False
    """True when graph ranking marks the claim as key."""

    topic_tags: list[str] = field(default_factory=list)
    """Thematic tags for the claim (e.g., 'Economy', 'War')."""

    def __post_init__(self) -> None:
        # Clamp check_worthiness to [0, 1]
        self.check_worthiness = max(0.0, min(1.0, self.check_worthiness))

        # Ensure nested objects are proper types
        if isinstance(self.search_locale_plan, dict):
            self.search_locale_plan = SearchLocalePlan(**self.search_locale_plan)
        if isinstance(self.retrieval_policy, dict):
            self.retrieval_policy = RetrievalPolicy(**self.retrieval_policy)

        if isinstance(self.time_signals, dict):
            self.time_signals = [self.time_signals]
        if isinstance(self.locale_signals, dict):
            self.locale_signals = [self.locale_signals]
        if self.time_signals is None:
            self.time_signals = []
        if self.locale_signals is None:
            self.locale_signals = []

    @property
    def should_skip_search(self) -> bool:
        """Check if search should be skipped for this claim."""
        return (
            self.verification_target == VerificationTarget.NONE and
            self.metadata_confidence != MetadataConfidence.LOW  # Fail-open overrides
        )

    @property
    def is_explain_only(self) -> bool:
        """Check if claim is explain-only (doesn't affect RGBA)."""
        return self.claim_role in {ClaimRole.CONTEXT, ClaimRole.META, ClaimRole.BACKGROUND}

    @property
    def role_weight(self) -> float:
        """Get RGBA aggregation weight based on role."""
        weights = {
            ClaimRole.CORE: 1.0,
            ClaimRole.SUPPORT: 0.8,
            ClaimRole.ATTRIBUTION: 0.7,
            ClaimRole.AGGREGATED: 0.6,
            ClaimRole.SUBCLAIM: 0.5,
            ClaimRole.CONTEXT: 0.0,  # Explain-only
            ClaimRole.META: 0.0,      # Explain-only
            ClaimRole.THESIS: 1.0,
            ClaimRole.BACKGROUND: 0.0,  # Explain-only
            ClaimRole.EXAMPLE: 0.5,
            ClaimRole.HEDGE: 0.2,
            ClaimRole.COUNTERCLAIM: 0.7,
        }
        # If verification_target is NONE, weight is 0 regardless of role
        if self.verification_target == VerificationTarget.NONE:
            return 0.0
        return weights.get(self.claim_role, 0.5)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/trace output."""
        return {
            "verification_target": self.verification_target.value,
            "claim_role": self.claim_role.value,
            "check_worthiness": self.check_worthiness,
            "is_key_claim": self.is_key_claim,
            "search_locale_plan": {
                "primary": self.search_locale_plan.primary,
                "fallback": self.search_locale_plan.fallback,
            },
            "time_signals": self.time_signals,
            "locale_signals": self.locale_signals,
            "time_sensitive": self.time_sensitive,
            "retrieval_policy": {
                "channels_allowed": [c.value for c in self.retrieval_policy.channels_allowed],
                "use_policy_by_channel": {
                    k: (v.value if isinstance(v, UsePolicy) else str(v))
                    for k, v in (self.retrieval_policy.use_policy_by_channel or {}).items()
                },
            },
            "metadata_confidence": self.metadata_confidence.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimMetadata":
        """Deserialize from dict."""
        if not data:
            return cls()

        # Parse verification_target
        vt_raw = data.get("verification_target", "reality")
        try:
            verification_target = VerificationTarget(vt_raw)
        except ValueError:
            verification_target = VerificationTarget.REALITY

        # Parse claim_role
        cr_raw = data.get("claim_role", "core")
        try:
            claim_role = ClaimRole(cr_raw)
        except ValueError:
            claim_role = ClaimRole.CORE

        # Parse search_locale_plan
        slp_raw = data.get("search_locale_plan", {})
        if isinstance(slp_raw, dict):
            search_locale_plan = SearchLocalePlan(
                primary=slp_raw.get("primary", "en"),
                fallback=slp_raw.get("fallback", ["en"]),
            )
        else:
            search_locale_plan = SearchLocalePlan()

        # Parse retrieval_policy
        rp_raw = data.get("retrieval_policy", {})
        if isinstance(rp_raw, dict):
            channels_raw = rp_raw.get("channels_allowed", [])
            channels = []
            for c in channels_raw:
                try:
                    # Backward compat: old payloads used "low_reliability"
                    cc = _normalize_channel_token(str(c))
                    if cc == "low_reliability":
                        cc = EvidenceChannel.LOW_RELIABILITY.value
                    channels.append(EvidenceChannel(cc))
                except ValueError:
                    pass

            use_policy_raw = rp_raw.get("use_policy_by_channel")
            if use_policy_raw is None:
                use_policy_raw = rp_raw.get("use_policy", {})

            retrieval_policy = RetrievalPolicy(
                channels_allowed=channels if channels else [
                    EvidenceChannel.AUTHORITATIVE,
                    EvidenceChannel.REPUTABLE_NEWS,
                ],
                use_policy=use_policy_raw if isinstance(use_policy_raw, dict) else {},
            )
        else:
            retrieval_policy = RetrievalPolicy()

        # Parse metadata_confidence
        mc_raw = data.get("metadata_confidence", "medium")
        try:
            metadata_confidence = MetadataConfidence(mc_raw)
        except ValueError:
            metadata_confidence = MetadataConfidence.MEDIUM

        time_signals_raw = data.get("time_signals", []) or []
        locale_signals_raw = data.get("locale_signals", []) or []
        if isinstance(time_signals_raw, dict):
            time_signals_raw = [time_signals_raw]
        if isinstance(locale_signals_raw, dict):
            locale_signals_raw = [locale_signals_raw]

        return cls(
            verification_target=verification_target,
            claim_role=claim_role,
            check_worthiness=float(data.get("check_worthiness", 0.5)),
            is_key_claim=bool(data.get("is_key_claim", False)),
            search_locale_plan=search_locale_plan,
            time_signals=[s for s in time_signals_raw if isinstance(s, dict)],
            locale_signals=[s for s in locale_signals_raw if isinstance(s, dict)],
            time_sensitive=bool(data.get("time_sensitive", False)),
            retrieval_policy=retrieval_policy,
            metadata_confidence=metadata_confidence,
            topic_tags=data.get("topic_tags") or [],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Default Factory
# ─────────────────────────────────────────────────────────────────────────────

def default_claim_metadata(
    *,
    verification_target: VerificationTarget = VerificationTarget.REALITY,
    confidence: MetadataConfidence = MetadataConfidence.LOW,
) -> ClaimMetadata:
    """
    Create default ClaimMetadata for fallback scenarios.
    
    When LLM fails to provide metadata, use this with LOW confidence
    to trigger fail-open behavior (Phase A-light always runs).
    """
    return ClaimMetadata(
        verification_target=verification_target,
        claim_role=ClaimRole.CORE,
        check_worthiness=0.5,
        is_key_claim=False,
        search_locale_plan=SearchLocalePlan(primary="en", fallback=["en"]),
        time_signals=[],
        locale_signals=[],
        time_sensitive=False,
        retrieval_policy=RetrievalPolicy(),
        metadata_confidence=confidence,
    )
