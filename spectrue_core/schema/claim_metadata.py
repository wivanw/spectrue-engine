"""
Claim-Centric Orchestration Metadata Types (Schema Adapter)

This module imports domain types and exposes them as schema types.
"""

from __future__ import annotations

from spectrue_core.domain.claims.model import (
    ClaimMetadata,
    ClaimRole,
    EvidenceChannel,
    MetadataConfidence,
    RetrievalPolicy,
    SearchLocalePlan,
    UsePolicy,
    VerificationTarget,
)
from spectrue_core.domain.claims.policy import default_claim_metadata

# Helper for backward compatibility or simple re-exports
# Ideally, consumers should import from domain, but schema is often the public contract.

__all__ = [
    "ClaimMetadata",
    "ClaimRole",
    "EvidenceChannel",
    "MetadataConfidence",
    "RetrievalPolicy",
    "SearchLocalePlan",
    "UsePolicy",
    "VerificationTarget",
    "default_claim_metadata",
]
