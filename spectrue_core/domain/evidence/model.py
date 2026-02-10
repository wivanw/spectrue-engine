"""Evidence domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class OracleStatus(str, Enum):
    """
    Oracle verdict status:
    - CONFIRMED: Fact-check confirms the claim is true
    - REFUTED: Fact-check says the claim is false/fake
    - MIXED: Fact-check says partially true or needs context
    - EMPTY: No relevant fact-check found
    - ERROR: API failure (check error_status_code)
    - DISABLED: Oracle validator not configured
    """
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    MIXED = "MIXED"
    EMPTY = "EMPTY"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


class ArticleIntent(str, Enum):
    """
    Article intent classification for Oracle triggering:
    - news: Current events, breaking news (CHECK Oracle)
    - evergreen: Science facts, historical claims, health info (CHECK Oracle)
    - official: Government/company announcements (CHECK Oracle)
    - opinion: Editorial, commentary (SKIP Oracle)
    - prediction: Future events (SKIP Oracle)
    """
    NEWS = "news"
    EVERGREEN = "evergreen"
    OFFICIAL = "official"
    OPINION = "opinion"
    PREDICTION = "prediction"


class OracleCheckResult(TypedDict, total=False):
    """
    Result from Google Fact Check API with LLM semantic validation.
    
    Used in hybrid Oracle flow:
    - JACKPOT (relevance > 0.9): Stop pipeline, return immediately
    - EVIDENCE (0.5 < relevance <= 0.9): Add to evidence pack, continue search
    - MISS (relevance <= 0.5 or EMPTY): Ignore, proceed to standard search
    """
    status: OracleStatus | str        # Verdict from fact-check
    url: str | None                   # URL of the fact-check article
    claim_reviewed: str | None        # The claim text from the external fact-check
    summary: str | None               # The verdict/explanation
    relevance_score: float            # 0.0 to 1.0 (Calculated by LLM)
    is_jackpot: bool                  # True if relevance > 0.9 (Stop search immediately)
    publisher: str | None             # Fact-check publisher name (Snopes, PolitiFact, etc.)
    rating: str | None                # Original textual rating from fact-checker
    source_provider: str | None       # UX: "Snopes via Google Fact Check"
    error_status_code: int | None     # HTTP status code on failure
    error_detail: str | None          # Error message


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


