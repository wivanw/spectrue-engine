"""Logic for deriving verdict states and labels from scores and evidence."""

from __future__ import annotations

from typing import Any


CANONICAL_VERDICT_STATES = {
    "supported",
    "refuted",
    "conflicted",
    "insufficient_evidence",
}


def count_stance_evidence(
    claim_id: str | None,
    evidence_items: list[dict[str, Any]],
) -> tuple[int, int, str | None]:
    """
    Count supporting and refuting evidence for a claim.
    """
    n_support = 0
    n_refute = 0
    best_tier = None
    
    # Deferred import to avoid circular dependencies
    from spectrue_core.domain.evidence.model import get_tier_rank
    
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        item_claim_id = item.get("claim_id")
        if claim_id and item_claim_id not in (None, claim_id):
            continue
        
        stance = str(item.get("stance") or "").lower()
        tier = item.get("tier")
        
        match stance:
            case "support" | "sup" | "supported":
                n_support += 1
            case "refute" | "ref" | "refuted":
                n_refute += 1
        
        # Track best tier
        if tier and (
            best_tier is None or get_tier_rank(tier) > get_tier_rank(best_tier)
        ):
            best_tier = tier
    
    return n_support, n_refute, best_tier


def derive_verdict_state_from_llm_score(
    llm_score: float,
    n_support: int,
    n_refute: int,
) -> str:
    """
    Derive canonical verdict state from LLM score and evidence counts.
    """
    if llm_score > 0.65:
        return "supported"
    elif llm_score < 0.35:
        return "refuted"
    elif n_support > 0 or n_refute > 0:
        return "conflicted"
    else:
        return "insufficient_evidence"


def derive_verdict_from_score(llm_score: float) -> str:
    """
    Derive verdict label from LLM score.
    """
    if llm_score > 0.65:
        return "verified"
    elif llm_score < 0.35:
        return "refuted"
    else:
        return "ambiguous"


def detect_evidence_conflict(n_support: int, n_refute: int) -> bool:
    """
    Detect if evidence is conflicting.
    """
    return n_support > 0 and n_refute > 0


def check_has_direct_evidence(
    claim_id: str | None,
    evidence_items: list[dict[str, Any]],
) -> bool:
    """
    Check if claim has direct evidence (SUPPORT/REFUTE with quote).
    """
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        
        if claim_id and item.get("claim_id") not in (None, claim_id):
            continue
        
        stance = str(item.get("stance") or "").upper()
        if stance in ("SUPPORT", "REFUTE") and item.get("quote"):
            return True
    
    return False
