"""Claim domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spectrue_core.domain.evidence.model import EvidenceChannel, UsePolicy


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

    DEFINITION = "definition"
    """Term definition or concept explanation."""

    FORECAST = "forecast"
    """Prediction or forecast about future events. Limited verifiability."""


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
    
    def get_use_policy(self, channel: EvidenceChannel | str) -> UsePolicy:
        """Get usage policy for a specific channel."""
        key = channel.value if isinstance(channel, EvidenceChannel) else str(channel)
        val = self.use_policy.get(key)
        if val is None:
             # Default: assume support is okay if allowed
             return UsePolicy.SUPPORT_OK
        if isinstance(val, UsePolicy):
             return val
        try:
             return UsePolicy(val)
        except ValueError:
             return UsePolicy.SUPPORT_OK
    # but simple getters on dataclass might be acceptable.
    # We strip logic for now.


@dataclass
class ClaimMetadata:
    """
    Complete metadata for claim-centric orchestration.
    
    This is the OUTPUT of claim extraction (filled by LLM).
    This is the INPUT to the orchestrator (builds ExecutionPlan).
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
            "topic_tags": self.topic_tags,
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


def _normalize_channel_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


