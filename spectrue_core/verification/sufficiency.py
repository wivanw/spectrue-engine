# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M80: Evidence Sufficiency Check

Determines if collected evidence is sufficient to make a verdict,
enabling early exit from progressive widening.

Sufficiency Rules:
- Rule 1: 1 authoritative source with quote (SUPPORT/REFUTE stance)
- Rule 2: 2 independent reputable sources with quotes (different domains)
- Rule 3: For attribution/existence: 1 origin source

Not Sufficient:
- Only social/low_reliability_web channels with lead_only policy
- Only CONTEXT stance sources (no SUPPORT/REFUTE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from spectrue_core.schema.claim_metadata import VerificationTarget, EvidenceChannel, UsePolicy
from spectrue_core.verification.source_utils import has_evidence_chunk
from spectrue_core.verification.trusted_sources import (
    TRUSTED_SOURCES,
    TIER_A_TLDS,
    TIER_A_SUFFIXES,
    ALL_TRUSTED_DOMAINS,
    is_social_platform,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sufficiency Result
# ─────────────────────────────────────────────────────────────────────────────

class SufficiencyStatus(str, Enum):
    """Result of sufficiency check."""
    SUFFICIENT = "sufficient"
    """Enough evidence to proceed with scoring."""
    
    INSUFFICIENT = "insufficient"
    """Need more evidence, continue widening."""
    
    SKIP = "skip"
    """No search needed (verification_target=none)."""


@dataclass
class SufficiencyResult:
    """
    Result of evidence sufficiency check for a claim.
    
    Contains status, reason, and metrics about the evidence.
    """
    claim_id: str
    """Claim being evaluated."""
    
    status: SufficiencyStatus = SufficiencyStatus.INSUFFICIENT
    """Whether evidence is sufficient."""
    
    reason: str = ""
    """Human-readable explanation for tracing."""
    
    rule_matched: str = ""
    """Which sufficiency rule was satisfied (Rule1, Rule2, Rule3, etc.)."""
    
    # Evidence metrics
    authoritative_count: int = 0
    """Count of authoritative (Tier A) sources."""
    
    reputable_count: int = 0
    """Count of reputable news (Tier B) sources."""
    
    independent_domains: int = 0
    """Count of unique domains with SUPPORT/REFUTE stance."""
    
    has_quotes: bool = False
    """Whether any source has a relevant quote."""
    
    support_refute_count: int = 0
    """Count of sources with SUPPORT or REFUTE stance."""
    
    context_only_count: int = 0
    """Count of sources with only CONTEXT stance."""
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for tracing."""
        return {
            "claim_id": self.claim_id,
            "status": self.status.value,
            "reason": self.reason,
            "rule_matched": self.rule_matched,
            "authoritative_count": self.authoritative_count,
            "reputable_count": self.reputable_count,
            "independent_domains": self.independent_domains,
            "has_quotes": self.has_quotes,
            "support_refute_count": self.support_refute_count,
            "context_only_count": self.context_only_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tier Detection Functions (T16)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract domain from URL, removing www prefix."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def is_authoritative(domain: str) -> bool:
    """
    T16: Check if domain is authoritative (Tier A).
    
    Tier A includes:
    - .gov, .edu, .mil, .int TLDs
    - Official sub-TLDs (.gov.uk, .europa.eu, etc.)
    - International public bodies (who.int, cdc.gov, nasa.gov)
    - Science and health institutions
    
    Args:
        domain: Domain string (without protocol)
        
    Returns:
        True if authoritative (Tier A)
    """
    if not domain:
        return False
    
    d = domain.lower().strip()
    
    # Check TLDs
    for tld in TIER_A_TLDS:
        if d.endswith(tld):
            return True
    
    # Check sub-TLDs
    for suffix in TIER_A_SUFFIXES:
        if d.endswith(suffix):
            return True
    
    # Check explicit lists
    if d in TRUSTED_SOURCES.get("international_public_bodies", []):
        return True
    if d in TRUSTED_SOURCES.get("science_and_health", []):
        return True
    if d in TRUSTED_SOURCES.get("astronomy_tier_a", []):
        return True
    if d in TRUSTED_SOURCES.get("global_news_agencies", []):
        return True
    if d in TRUSTED_SOURCES.get("fact_checking_ifcn", []):
        return True
    
    return False


def is_reputable_news(domain: str) -> bool:
    """
    T16: Check if domain is reputable news (Tier B).
    
    Tier B includes:
    - Major news outlets
    - Regional quality media
    - Trusted editorial sources
    
    Args:
        domain: Domain string
        
    Returns:
        True if reputable news (Tier B)
    """
    if not domain:
        return False
    
    d = domain.lower().strip()
    
    # Check against all trusted domains
    if d in ALL_TRUSTED_DOMAINS:
        return True
    
    # Check specific categories
    tier_b_categories = [
        "general_news_western",
        "ukraine_imi_whitelist",
        "general_news_ukraine_broad",
        "europe_tier1",
        "asia_pacific",
        "technology",
        "astronomy_tier_b",
        "russia_independent_exiled",
    ]
    
    for category in tier_b_categories:
        if d in TRUSTED_SOURCES.get(category, []):
            return True
    
    return False


def is_local_media(domain: str) -> bool:
    """
    Check if domain is local/regional media (Tier C).
    
    Not in Tier A or B but has legitimate news structure.
    """
    if not domain:
        return False
    
    d = domain.lower().strip()
    
    # Not social and not authoritative/reputable
    if is_social_platform(d):
        return False
    if is_authoritative(d) or is_reputable_news(d):
        return False
    
    # Assume local media if it has news-like TLD patterns
    news_patterns = [".news", ".media", ".tv", ".radio", ".times", ".post", ".herald"]
    for pattern in news_patterns:
        if pattern in d:
            return True
    
    return False


def get_domain_tier(domain: str) -> EvidenceChannel:
    """
    Determine the evidence channel for a domain.
    
    Args:
        domain: Domain string
        
    Returns:
        EvidenceChannel enum value
    """
    if not domain:
        return EvidenceChannel.LOW_RELIABILITY
    
    if is_authoritative(domain):
        return EvidenceChannel.AUTHORITATIVE
    
    if is_reputable_news(domain):
        return EvidenceChannel.REPUTABLE_NEWS
    
    if is_social_platform(domain):
        return EvidenceChannel.SOCIAL
    
    if is_local_media(domain):
        return EvidenceChannel.LOCAL_MEDIA
    
    return EvidenceChannel.LOW_RELIABILITY


def is_origin_source(source: Any, claim_text: str) -> bool:
    """
    T16: Check if source is the origin of an attribution claim.
    
    For claims like "Company X announced Y", the origin source
    would be company X's official website.
    
    Args:
        source: Source dict with url, content, etc.
        claim_text: The claim text (normalized)
        
    Returns:
        True if this appears to be the origin source
    """
    # Fail-closed: only structured sources are eligible for "origin" promotion.
    # Some providers/tools may emit plain URL strings; those should never satisfy
    # Rule 3 (origin) because they lack attribution/quote/title signals.
    if not isinstance(source, dict):
        return False

    url = source.get("url", "") or source.get("link", "")
    domain = _extract_domain(url)
    
    if not domain:
        return False
    
    # For now, simple heuristic: official domains for attributed statements
    # TODO: Enhance with NER to extract entity and match to domain
    
    # Check if authoritative (official sources are often origins)
    if is_authoritative(domain):
        return True
    
    # Check if source has "official" in title or is primary source
    title = (source.get("title", "") or "").lower()
    if "official" in title or "statement" in title or "announcement" in title:
        return True
    
    # Check if source is marked as primary
    if source.get("is_primary") or source.get("is_origin"):
        return True
    
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Sufficiency Check (T14, T15)
# ─────────────────────────────────────────────────────────────────────────────

def evidence_sufficiency(
    claim_id: str,
    sources: list[Any],
    verification_target: VerificationTarget = VerificationTarget.REALITY,
    claim_text: str = "",
    use_policy_by_channel: dict[str, Any] | None = None,
) -> SufficiencyResult:
    """
    T14/T15: Check if evidence is sufficient for a claim.
    
    Sufficient if ANY rule is satisfied:
    - Rule 1: 1 authoritative source with quote (SUPPORT/REFUTE)
    - Rule 2: 2 independent reputable sources with quotes (different domains)
    - Rule 3: For attribution/existence - 1 origin source
    
    Not sufficient if:
    - Only social/low_reliability_web sources (lead-only)
    - Only CONTEXT stance (no SUPPORT/REFUTE)
    
    Args:
        claim_id: The claim being checked
        sources: List of source dicts with url, stance, etc.
        verification_target: What we're trying to verify
        claim_text: Claim text for origin matching
        
    Returns:
        SufficiencyResult with status and metrics
    """
    result = SufficiencyResult(claim_id=claim_id)
    
    # Skip check for non-verifiable claims
    if verification_target == VerificationTarget.NONE:
        result.status = SufficiencyStatus.SKIP
        result.reason = "verification_target=none, no search needed"
        return result
    
    if not sources:
        result.status = SufficiencyStatus.INSUFFICIENT
        result.reason = "No sources available"
        return result
    
    # Analyze sources
    authoritative_sources: list[dict] = []
    reputable_sources: list[dict] = []
    origin_sources: list[dict] = []
    support_refute_domains: set[str] = set()
    has_quotes = False
    context_only = True
    
    # Normalize use_policy_by_channel values to UsePolicy.
    policy_map: dict[str, UsePolicy] = {}
    if isinstance(use_policy_by_channel, dict):
        for k, v in use_policy_by_channel.items():
            try:
                if isinstance(v, UsePolicy):
                    policy_map[str(k)] = v
                else:
                    policy_map[str(k)] = UsePolicy(str(v))
            except Exception:
                continue

    lead_only_count = 0

    for source in sources:
        if isinstance(source, str):
            url = source
            stance = ""
            has_quote = False
        else:
            url = source.get("url", "") or source.get("link", "")
            stance = (source.get("stance", "") or "").lower()
            has_quote = bool(source.get("quote") or source.get("snippet") or source.get("content"))
        
        if not url:
            continue
            
        domain = _extract_domain(url)
        
        # Track quotes
        if has_quote:
            has_quotes = True
        
        # Track stance
        if stance in ("support", "refute", "contradict"):
            context_only = False
            result.support_refute_count += 1
            if domain:
                support_refute_domains.add(domain)
        else:
            result.context_only_count += 1
        
        # Categorize by tier
        tier = get_domain_tier(domain)

        # Enforce lead_only policy for social/low-reliability: these are leads, not evidence.
        if tier in (EvidenceChannel.SOCIAL, EvidenceChannel.LOW_RELIABILITY):
            policy = policy_map.get(tier.value, UsePolicy.LEAD_ONLY)
            if policy == UsePolicy.LEAD_ONLY:
                lead_only_count += 1
                # Do not count toward sufficiency rules; keep tracking for diagnostics.
                continue
        
        if tier == EvidenceChannel.AUTHORITATIVE:
            result.authoritative_count += 1
            if stance in ("support", "refute", "contradict") and has_quote:
                authoritative_sources.append(source)
        
        elif tier in (EvidenceChannel.REPUTABLE_NEWS, EvidenceChannel.LOCAL_MEDIA):
            result.reputable_count += 1
            if stance in ("support", "refute", "contradict") and has_quote:
                reputable_sources.append(source)
        
        # Check for origin source (attribution/existence claims).
        # Origin requires a real evidence chunk (quote/snippet/content), not just a bare link.
        if verification_target in (VerificationTarget.ATTRIBUTION, VerificationTarget.EXISTENCE):
            if has_evidence_chunk(source) and is_origin_source(source, claim_text):
                origin_sources.append(source)
    
    result.independent_domains = len(support_refute_domains)
    result.has_quotes = has_quotes
    
    # ─────────────────────────────────────────────────────────────────────────
    # Apply Sufficiency Rules
    # ─────────────────────────────────────────────────────────────────────────
    
    # Rule 3: For attribution/existence, 1 origin source is enough
    if verification_target in (VerificationTarget.ATTRIBUTION, VerificationTarget.EXISTENCE):
        if len(origin_sources) >= 1:
            result.status = SufficiencyStatus.SUFFICIENT
            result.rule_matched = "Rule3"
            result.reason = f"Found {len(origin_sources)} origin source(s) for {verification_target.value} claim"
            return result
    
    # Rule 1: 1 authoritative source with quote
    if len(authoritative_sources) >= 1:
        result.status = SufficiencyStatus.SUFFICIENT
        result.rule_matched = "Rule1"
        result.reason = f"Found {len(authoritative_sources)} authoritative source(s) with quotes"
        return result
    
    # Rule 2: 2 independent reputable sources with quotes
    # Check domain independence
    reputable_domains: set[str] = set()
    for source in reputable_sources:
        url = source.get("url", "") or source.get("link", "")
        domain = _extract_domain(url)
        if domain:
            reputable_domains.add(domain)
    
    if len(reputable_domains) >= 2:
        result.status = SufficiencyStatus.SUFFICIENT
        result.rule_matched = "Rule2"
        result.reason = f"Found {len(reputable_domains)} independent reputable sources with quotes"
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Not Sufficient - Determine Reason
    # ─────────────────────────────────────────────────────────────────────────
    
    result.status = SufficiencyStatus.INSUFFICIENT
    
    if lead_only_count > 0 and (lead_only_count == len(sources)):
        result.reason = "All sources are lead-only (social/low_reliability_web); no usable evidence chunks"
    elif context_only:
        result.reason = "All sources have CONTEXT stance only, no SUPPORT/REFUTE"
    elif result.authoritative_count == 0 and result.reputable_count < 2:
        result.reason = f"Need authoritative or 2+ reputable sources. Have: {result.authoritative_count} auth, {result.reputable_count} reputable"
    elif not has_quotes:
        result.reason = "Sources lack quotes/content"
    else:
        result.reason = f"Insufficient quality: {result.independent_domains} independent domains"
    
    return result


def check_sufficiency_for_claim(
    claim: dict,
    sources: list[Any],
) -> SufficiencyResult:
    """
    Convenience wrapper that extracts metadata from claim.
    
    Args:
        claim: Claim dict with id, metadata, normalized_text
        sources: List of sources for this claim
        
    Returns:
        SufficiencyResult
    """
    claim_id = claim.get("id", "unknown")
    claim_text = claim.get("normalized_text", "") or claim.get("text", "")
    
    metadata = claim.get("metadata")
    if metadata:
        verification_target = metadata.verification_target
        use_policy_by_channel = getattr(metadata.retrieval_policy, "use_policy_by_channel", {}) if getattr(metadata, "retrieval_policy", None) else {}
    else:
        verification_target = VerificationTarget.REALITY
        use_policy_by_channel = {}
    
    return evidence_sufficiency(
        claim_id=claim_id,
        sources=sources,
        verification_target=verification_target,
        claim_text=claim_text,
        use_policy_by_channel=use_policy_by_channel,
    )
