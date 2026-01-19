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
Evidence Sufficiency Check

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
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

from spectrue_core.schema.claim_metadata import VerificationTarget, EvidenceChannel
from spectrue_core.tools.trusted_sources import get_domain_tier, is_authoritative
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spectrue_core.verification.search.search_policy import SearchPolicyProfile


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


class SufficiencyDecision(str, Enum):
    """High-level decision for iterative retrieval."""
    ENOUGH = "ENOUGH"
    NEED_FOLLOWUP = "NEED_FOLLOWUP"
    STOP = "STOP"


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


# Bayesian Parameters (Scientific Scoring)
BASE_PRIOR_P: float = 0.5       # Neutral starting point (0.0 log-odds)
SUFFICIENCY_P_THRESHOLD: float = 0.95
sufficiency_threshold = SUFFICIENCY_P_THRESHOLD  # Alias for backward compatibility

# Tier-to-Support-Probability Mapping (Veracity Support Signals)
# These represent the probability that a "hit" in this tier is factually correct.
# Used as Bayes Factors for search sufficiency.
TIER_SUPPORT_PROBABILITIES = {
    EvidenceChannel.AUTHORITATIVE: 0.96,    # Tier A
    EvidenceChannel.REPUTABLE_NEWS: 0.90,   # Tier B (Wikipedia now here)
    EvidenceChannel.LOCAL_MEDIA: 0.75,      # Tier C
    EvidenceChannel.SOCIAL: 0.55,           # Tier D (Weak signal)
    EvidenceChannel.LOW_RELIABILITY: 0.52, # Very weak
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

    # Priority 1: Accept explicit origin signals from structured metadata.
    source_type = str(source.get("source_type") or "").lower()
    if source_type in {"primary", "official", "fact_check"}:
        return True
    if source.get("is_primary") or source.get("is_origin"):
        return True

    # Priority 2: Authoritative domains (official sources are often origins)
    if is_authoritative(domain):
        return True

    # Priority 3: Title-based heuristics for official statements
    title = (source.get("title", "") or "").lower()
    if "official" in title or "statement" in title or "announcement" in title:
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
    Bayesian Sufficiency Judge.
    
    Accumulates evidence signals in log-odds space based on source tiers.
    Replaces heuristic Rules with a unified probabilistic threshold.
    """
    from spectrue_core.scoring.belief import prob_to_log_odds, log_odds_to_prob

    result = SufficiencyResult(claim_id=claim_id)

    if verification_target == VerificationTarget.NONE:
        result.status = SufficiencyStatus.SKIP
        result.reason = "verification_target=none, no search needed"
        return result

    if not sources:
        result.status = SufficiencyStatus.INSUFFICIENT
        result.reason = "No sources available"
        return result

    # 1. Collect unique domains to avoid over-inflating from one site
    # 2. Assign best tier for each domain
    domain_best_tier: dict[str, EvidenceChannel] = {}
    domain_has_quote: dict[str, bool] = {}
    
    for src in sources:
        url = src.get("url") or src.get("link") if isinstance(src, dict) else str(src)
        if not url: continue
        
        domain = _extract_domain(url)
        tier = get_domain_tier(domain)
        
        # Track best tier per domain
        if domain not in domain_best_tier:
            domain_best_tier[domain] = tier
        else:
            # If we found it on a higher tier (unlikely for same domain but safe)
            current_rank = {"A": 4, "B": 3, "C": 2, "D": 1}.get(tier.value[0], 0) if hasattr(tier, "value") else 0
            best_rank = {"A": 4, "B": 3, "C": 2, "D": 1}.get(domain_best_tier[domain].value[0], 0)
            if current_rank > best_rank:
                domain_best_tier[domain] = tier
        
        # Track if domain provided a quote
        if isinstance(src, dict) and src.get("quote"):
            domain_has_quote[domain] = True
            result.has_quotes = True

    # 3. Accumulate Log-Odds
    # Formula: L_post = L_prior + sum( logit(P_tier_i) )
    total_log_odds = prob_to_log_odds(BASE_PRIOR_P)
    
    for domain, tier in domain_best_tier.items():
        p_support = TIER_SUPPORT_PROBABILITIES.get(tier, 0.5)
        
        # Penalize if no quote (weakens the signal)
        if not domain_has_quote.get(domain):
            p_support = 0.5 + (p_support - 0.5) * 0.3  # Reduce weight significantly if snippet only
            
        logv = prob_to_log_odds(p_support)
        total_log_odds += logv
        
        # Update metrics for diagnostics
        if tier == EvidenceChannel.AUTHORITATIVE: result.authoritative_count += 1
        if tier == EvidenceChannel.REPUTABLE_NEWS: result.reputable_count += 1
        
    result.independent_domains = len(domain_best_tier)
    posterior_p = log_odds_to_prob(total_log_odds)
    
    # 4. Check Against Threshold
    if posterior_p >= SUFFICIENCY_P_THRESHOLD:
        result.status = SufficiencyStatus.SUFFICIENT
        result.rule_matched = "BayesianConsensus"
        result.reason = f"Combined confidence {posterior_p:.1%} >= {SUFFICIENCY_P_THRESHOLD:.1%} (domains: {len(domain_best_tier)})"
    else:
        result.status = SufficiencyStatus.INSUFFICIENT
        result.reason = f"Confidence {posterior_p:.1%} < {SUFFICIENCY_P_THRESHOLD:.1%}"

    return result


def verdict_ready_for_claim(
    sources: list[Any],
    *,
    claim_id: str = "",
) -> tuple[bool, dict]:
    """
    Determine whether evidence is strong enough to score a claim.

    Ready when:
    - at least one source for this claim with SUPPORT/REFUTE stance and quote_matches == True.
    """
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
            continue  # Only count anchors for SUPPORT/REFUTE

        if stance == "SUPPORT":
            stance_support += 1
        else:
            stance_refute += 1

        has_quote_matches = bool(src.get("quote_matches"))
        if has_quote_matches:
            quote_matches += 1
        if has_quote_matches:
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


def _estimate_coverage(sources: list[Any], min_relevance_score: float) -> float:
    if not sources:
        return 0.0
    hits = 0
    for src in sources:
        if not isinstance(src, dict):
            continue
        score = src.get("relevance_score", 0.0)
        try:
            if float(score or 0.0) >= float(min_relevance_score):
                hits += 1
        except Exception:
            continue
    return hits / len(sources)


def _estimate_diversity(sources: list[Any]) -> float:
    if not sources:
        return 0.0
    domains = set()
    for src in sources:
        if not isinstance(src, dict):
            continue
        url = src.get("url", "") or src.get("link", "")
        domain = _extract_domain(str(url))
        if domain:
            domains.add(domain)
    return len(domains) / len(sources)


def judge_sufficiency_for_claim(
    claim: dict,
    sources: list[Any],
    *,
    policy_profile: "SearchPolicyProfile | None" = None,
    use_policy_by_channel: dict[str, Any] | None = None,
) -> SufficiencyDecisionResult:
    """
    Decide whether to stop or continue iterative retrieval.

    Uses sufficiency rules, then applies policy thresholds to determine
    ENOUGH vs NEED_FOLLOWUP. Falls back to STOP on errors.
    """
    claim_id = claim.get("id", "unknown") if isinstance(claim, dict) else "unknown"

    try:
        metadata = claim.get("metadata") if isinstance(claim, dict) else getattr(claim, "metadata", None)
        if use_policy_by_channel is None:
            use_policy_by_channel = (
                getattr(metadata.retrieval_policy, "use_policy_by_channel", {})
                if getattr(metadata, "retrieval_policy", None)
                else {}
            )

        verification_target = VerificationTarget.REALITY
        if metadata and getattr(metadata, "verification_target", None):
            verification_target = metadata.verification_target

        sufficiency = evidence_sufficiency(
            claim_id=claim_id,
            sources=sources,
            verification_target=verification_target,
            claim_text=claim.get("normalized_text", "") or claim.get("text", ""),
            use_policy_by_channel=use_policy_by_channel,
        )

        # T003: Removed 'context stance' from degraded logic because we no longer 
        # check for stance during retrieval. 'lack usable evidence' replaces it.
        degraded = "lead-only" in sufficiency.reason.lower() or "lack usable evidence" in sufficiency.reason.lower()

        if sufficiency.status == SufficiencyStatus.SKIP:
            decision = SufficiencyDecision.STOP
        elif sufficiency.status == SufficiencyStatus.SUFFICIENT:
            decision = SufficiencyDecision.ENOUGH
        else:
            decision = SufficiencyDecision.NEED_FOLLOWUP

        coverage = 0.0
        diversity = 0.0
        aligned_hits = 0
        if policy_profile is not None and decision != SufficiencyDecision.STOP:
            thresholds = policy_profile.quality_thresholds
            coverage = _estimate_coverage(sources, thresholds.min_relevance_score)
            diversity = _estimate_diversity(sources)
            aligned_hits = sum(
                1
                for src in sources
                if isinstance(src, dict)
                and float(src.get("relevance_score") or 0.0) >= float(thresholds.min_relevance_score)
                and str(src.get("claim_id") or claim_id) == str(claim_id)
            )
            if (
                coverage < thresholds.min_coverage
                or diversity < thresholds.min_diversity
                or aligned_hits == 0
            ):
                decision = SufficiencyDecision.NEED_FOLLOWUP
                sufficiency.reason = (
                    f"{sufficiency.reason}; below quality thresholds "
                    f"(coverage={coverage:.2f}, diversity={diversity:.2f}, aligned_hits={aligned_hits})"
                )

            if not policy_profile.stop_conditions.stop_on_sufficiency and decision == SufficiencyDecision.ENOUGH:
                decision = SufficiencyDecision.NEED_FOLLOWUP
                sufficiency.reason = "Policy requires follow-up despite sufficiency"

        return SufficiencyDecisionResult(
            claim_id=claim_id,
            decision=decision,
            reason=sufficiency.reason,
            rule_matched=sufficiency.rule_matched,
            degraded_confidence=degraded,
            coverage=coverage,
            diversity=diversity,
        )
    except Exception as exc:
        logger.warning("[M92] Sufficiency judge failed for %s: %s", claim_id, exc)
        return SufficiencyDecisionResult(
            claim_id=claim_id,
            decision=SufficiencyDecision.STOP,
            reason="Sufficiency judge failed",
            degraded_confidence=True,
        )
