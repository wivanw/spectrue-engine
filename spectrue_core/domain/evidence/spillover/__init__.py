"""Evidence spillover package."""

from .models import (
    SpilloverChoice, SpilloverResult, NormalizeUrl,
    SlotsFromAssertionKey, RequiredSlotsForTarget, MergeCovers,
    ClaimEventSignature, EvidenceEventSignature, SignatureCompatible
)
from .ranking import score_for_transfer, claim_topic_signature, topic_overlap_boost
from .compute import compute_spillover

__all__ = [
    "SpilloverChoice",
    "SpilloverResult",
    "NormalizeUrl",
    "SlotsFromAssertionKey",
    "RequiredSlotsForTarget",
    "MergeCovers",
    "ClaimEventSignature",
    "EvidenceEventSignature",
    "SignatureCompatible",
    "score_for_transfer",
    "claim_topic_signature",
    "topic_overlap_boost",
    "compute_spillover",
]
