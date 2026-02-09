"""Claim policy logic."""

from __future__ import annotations

from spectrue_core.domain.claims.model import (
    ClaimMetadata,
    ClaimRole,
    VerificationTarget,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
)
from spectrue_core.domain.evidence.model import EvidenceChannel, UsePolicy
from enum import Enum
from dataclasses import dataclass
class PolicyMode(str, Enum):
    """Per-claim policy decision for search routing."""
    SKIP = "SKIP"
    CHEAP = "CHEAP"
    FULL = "FULL"


@dataclass(frozen=True)
class ClaimPolicyDecision:
    """Routing decision result for a claim before query building."""
    mode: PolicyMode
    reason_codes: list[str]



def get_role_weight(metadata: ClaimMetadata) -> float:
    """
    Get RGBA aggregation weight based on role.
    
    Weights determine how much the claim contributes to the overall document score.
    """
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
        ClaimRole.DEFINITION: 0.3,
        ClaimRole.FORECAST: 0.3,  # Limited verifiability
    }
    # If verification_target is NONE, weight is 0 regardless of role
    if metadata.verification_target == VerificationTarget.NONE:
        return 0.0
    return weights.get(metadata.claim_role, 0.5)


def should_skip_search(metadata: ClaimMetadata) -> bool:
    """
    Check if search should be skipped for this claim.
    """
    return (
        metadata.verification_target == VerificationTarget.NONE and
        metadata.metadata_confidence != MetadataConfidence.LOW  # Fail-open overrides
    )


def is_explain_only(metadata: ClaimMetadata) -> bool:
    """
    Check if claim is explain-only (doesn't affect RGBA).
    """
    return metadata.claim_role in {
        ClaimRole.CONTEXT, 
        ClaimRole.META, 
        ClaimRole.BACKGROUND
    }


def retrieval_get_use_policy(policy: RetrievalPolicy, channel: EvidenceChannel) -> UsePolicy:
    """Get usage policy for a channel. Defaults to LEAD_ONLY."""
    val = policy.use_policy.get(channel.value, UsePolicy.LEAD_ONLY.value)
    return UsePolicy(val)


def retrieval_can_support(policy: RetrievalPolicy, channel: EvidenceChannel) -> bool:
    """Check if channel can provide supporting evidence."""
    return retrieval_get_use_policy(policy, channel) == UsePolicy.SUPPORT_OK


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
