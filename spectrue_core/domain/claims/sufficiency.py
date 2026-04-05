from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from spectrue_core.domain.claims.model import (
    VerificationTarget,
    EvidenceChannel,
)
from spectrue_core.tools.trusted_sources import get_domain_tier
from spectrue_core.domain.verification.verdict.belief import prob_to_log_odds, log_odds_to_prob

logger = logging.getLogger(__name__)

# Bayesian Evidence Accumulation Constants
BASE_PRIOR_P = 0.5  # Neutral prior
SUFFICIENCY_P_THRESHOLD = 0.82  # Threshold for "enough" evidence (posterior P)

# Per-claim-type sufficiency thresholds (overrides SUFFICIENCY_P_THRESHOLD).
_SUFFICIENCY_BY_TYPE: dict[str, float] = {
    "core": 0.82,
    "numeric": 0.85,        # needs authoritative data source
    "timeline": 0.80,       # events well-covered by news
    "attribution": 0.78,    # finding the original source is often enough
    "sidefact": 0.70,       # low-value claims, don't waste searches
}

# Probability that a single independent source of a given tier 
# would support the claim if it were true.
TIER_SUPPORT_PROBABILITIES = {
    EvidenceChannel.AUTHORITATIVE: 0.95,   # Tier A
    EvidenceChannel.REPUTABLE_NEWS: 0.80,  # Tier B
    EvidenceChannel.LOCAL_MEDIA: 0.70,     # Tier C
    EvidenceChannel.SOCIAL: 0.60,          # Tier D
}


# ─────────────────────────────────────────────────────────────────────────────
# Sufficiency Types (Domain)
# ─────────────────────────────────────────────────────────────────────────────

class SufficiencyStatus(str, Enum):
    """Result of sufficiency check."""
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    SKIP = "skip"


class SufficiencyDecision(str, Enum):
    """High-level decision for iterative retrieval."""
    ENOUGH = "ENOUGH"
    NEED_FOLLOWUP = "NEED_FOLLOWUP"
    STOP = "STOP"


@dataclass
class SufficiencyResult:
    """Result of evidence sufficiency check for a claim."""
    claim_id: str
    status: SufficiencyStatus = SufficiencyStatus.INSUFFICIENT
    reason: str = ""
    rule_matched: str = ""
    authoritative_count: int = 0
    reputable_count: int = 0
    independent_domains: int = 0
    has_quotes: bool = False
    support_refute_count: int = 0
    context_only_count: int = 0
    posterior_p: float = 0.5


@dataclass
class SufficiencyDecisionResult:
    """Decision returned by the sufficiency judge."""
    claim_id: str
    decision: SufficiencyDecision
    reason: str
    rule_matched: str = ""
    degraded_confidence: bool = False
    coverage: float = 0.0
    diversity: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Domain Logic
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


def evidence_sufficiency(
    claim_id: str,
    sources: list[Any],
    verification_target: VerificationTarget = VerificationTarget.REALITY,
    claim_text: str = "",
    use_policy_by_channel: dict[str, Any] | None = None,
    claim_type: str = "core",
) -> SufficiencyResult:
    """
    Bayesian Sufficiency Judge (Domain).
    
    Accumulates evidence signals in log-odds space based on source tiers.
    """
    result = SufficiencyResult(claim_id=claim_id)

    if verification_target == VerificationTarget.NONE:
        result.status = SufficiencyStatus.SKIP
        result.reason = "verification_target=none, no search needed"
        return result

    if not sources:
        result.status = SufficiencyStatus.INSUFFICIENT
        result.reason = "No sources available"
        return result

    domain_best_tier: dict[str, EvidenceChannel] = {}
    domain_has_quote: dict[str, bool] = {}
    domain_best_stance: dict[str, str] = {}

    for src in sources:
        url = src.get("url") or src.get("link") if isinstance(src, dict) else str(src)
        if not url:
            continue

        domain = _extract_domain(url)
        tier = get_domain_tier(domain)

        if domain not in domain_best_tier:
            domain_best_tier[domain] = tier
        else:
            current_rank = {"A": 4, "B": 3, "C": 2, "D": 1}.get(tier.value[0], 0) if hasattr(tier, "value") else 0
            best_rank = {"A": 4, "B": 3, "C": 2, "D": 1}.get(domain_best_tier[domain].value[0], 0)
            if current_rank > best_rank:
                domain_best_tier[domain] = tier

        if isinstance(src, dict) and src.get("quote"):
            domain_has_quote[domain] = True
            result.has_quotes = True

        # Track best stance per domain (support > refute > other)
        if isinstance(src, dict):
            stance = str(src.get("stance") or "").lower()
            prev = domain_best_stance.get(domain, "")
            if stance == "support" or (stance == "refute" and prev != "support"):
                domain_best_stance[domain] = stance

    total_log_odds = prob_to_log_odds(BASE_PRIOR_P)
    n_domains = len(domain_best_tier)

    for domain, tier in domain_best_tier.items():
        p_support = TIER_SUPPORT_PROBABILITIES.get(tier, 0.5)

        # If no quote, reduce signal significantly (Spec Kit principle)
        if not domain_has_quote.get(domain):
            p_support = 0.5 + (p_support - 0.5) * 0.3

        # Stance-aware adjustment — only when multiple independent domains
        # contribute (a single source confirming itself is not informative).
        if n_domains >= 2:
            stance = domain_best_stance.get(domain, "")
            if stance == "support":
                p_support = min(0.99, p_support * 1.1)
            elif stance == "refute":
                p_support = max(0.5, p_support * 0.7)

        logv = prob_to_log_odds(p_support)
        total_log_odds += logv

        if tier == EvidenceChannel.AUTHORITATIVE:
            result.authoritative_count += 1
        if tier == EvidenceChannel.REPUTABLE_NEWS:
            result.reputable_count += 1

    result.independent_domains = n_domains
    posterior_p = log_odds_to_prob(total_log_odds)

    threshold = _SUFFICIENCY_BY_TYPE.get(claim_type, SUFFICIENCY_P_THRESHOLD)
    result.posterior_p = posterior_p
    if posterior_p >= threshold:
        result.status = SufficiencyStatus.SUFFICIENT
        result.rule_matched = "BayesianConsensus"
        result.reason = f"Combined confidence {posterior_p:.1%} >= {threshold:.1%} (type={claim_type}, domains: {len(domain_best_tier)})"
    else:
        result.status = SufficiencyStatus.INSUFFICIENT
        result.reason = f"Confidence {posterior_p:.1%} < {threshold:.1%} (type={claim_type})"

    return result


def verdict_ready_for_claim(
    sources: list[Any],
    *,
    claim_id: str = "",
) -> tuple[bool, dict]:
    """
    Determine whether evidence is strong enough to score a claim.
    """
    from spectrue_core.utils.trace import Trace
    
    anchor_count = 0
    matched_claim_id = 0
    stance_support = 0
    stance_refute = 0
    quote_matches = 0
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        src_claim_id = str(src.get("claim_id") or "")
        if claim_id and src_claim_id != claim_id:
            continue
        matched_claim_id += 1

        stance = str(src.get("stance") or "").upper()
        if stance not in {"SUPPORT", "REFUTE"}:
            continue

        if stance == "SUPPORT":
            stance_support += 1
        else:
            stance_refute += 1

        has_quote_matches = bool(src.get("quote_matches"))
        if has_quote_matches:
            quote_matches += 1
            anchor_count += 1

    ready = anchor_count >= 1
    stats = {
        "matched_claim_id": matched_claim_id,
        "stance_support": stance_support,
        "stance_refute": stance_refute,
        "quote_matches": quote_matches,
        "anchors": anchor_count,
    }
    Trace.event(
        "verdict.ready",
        {
            "claim_id": claim_id,
            "ready": ready,
            "stats": stats,
        },
    )
    return ready, stats


def is_origin_source(source: Any, claim_text: str) -> bool:
    """
    Checks if a source is likely the 'origin' source for an attribution claim.
    """
    if not source or not claim_text:
        return False
    
    if isinstance(source, dict):
        if source.get("is_primary"):
            return True
        title = str(source.get("title") or "").lower()
        if "official" in title and "statement" in title:
            return True
        url = source.get("url") or source.get("link")
    else:
        url = str(source)
        
    if not url:
        return False
        
    domain = _extract_domain(url)
    if not domain:
        return False
        
    # Extract name from domain (e.g. cdc.gov -> cdc)
    name = domain.split('.')[0]
    if len(name) >= 2 and name in claim_text.lower():
        return True
        
    return domain.lower() in claim_text.lower()


def check_sufficiency_for_claim(
    claim: dict,
    sources: list[Any],
) -> SufficiencyResult:
    """
    Convenience wrapper that extracts metadata from claim.
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

    claim_type = claim.get("type") or claim.get("claim_type") or "core"

    return evidence_sufficiency(
        claim_id=claim_id,
        sources=sources,
        verification_target=verification_target,
        claim_text=claim_text,
        use_policy_by_channel=use_policy_by_channel,
        claim_type=str(claim_type),
    )
