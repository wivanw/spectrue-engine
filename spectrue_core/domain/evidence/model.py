"""Evidence domain models."""

from __future__ import annotations

from enum import Enum


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
