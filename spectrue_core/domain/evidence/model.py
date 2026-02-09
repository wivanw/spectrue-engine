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


def norm_claim_id(x: Any) -> str | None:
    """Normalize claim ID for consistent lookup."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null", "undefined"):
        return None
    return s


def get_tier_rank(tier: str | None) -> int:
    """
    Get numeric rank for evidence tier.
    """
    if not tier:
        return 0
    return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(
        str(tier).strip().upper(), 0
    )


def find_best_tier_for_claim(
    claim_id: str | None,
    evidence_items: list[dict[str, Any]],
) -> str | None:
    """
    Find best (highest-ranked) evidence tier for a claim.
    """
    best_tier = None
    
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        item_claim_id = item.get("claim_id")
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        tier = item.get("tier")
        if tier and (
            best_tier is None or get_tier_rank(tier) > get_tier_rank(best_tier)
        ):
            best_tier = tier
    
    return best_tier


from typing import Any
