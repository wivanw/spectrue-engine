from __future__ import annotations

from typing import Any
import math


def extract_evidence_features(evidence_index: Any, claims: list[dict[str, Any]]) -> dict[str, float]:
    """Extract features from EvidenceIndex (includes global_pack for standard mode)."""
    total_items = 0
    n_with_stance = 0
    n_high_tier = 0
    n_with_quote = 0

    # Collect from by_claim_id
    for pack in getattr(evidence_index, "by_claim_id", {}).values():
        for item in getattr(pack, "items", ()) or ():
            total_items += 1
            if (getattr(item, "stance", "") or "").upper() in {"SUPPORT", "REFUTE"}:
                n_with_stance += 1
            if (getattr(item, "tier", "") or "").upper() in {"A", "A'", "B"}:
                n_high_tier += 1
            if getattr(item, "quote", None):
                n_with_quote += 1

    # Also collect from global_pack (standard mode)
    global_pack = getattr(evidence_index, "global_pack", None)
    if global_pack:
        for item in getattr(global_pack, "items", ()) or ():
            total_items += 1
            if (getattr(item, "stance", "") or "").upper() in {"SUPPORT", "REFUTE"}:
                n_with_stance += 1
            if (getattr(item, "tier", "") or "").upper() in {"A", "A'", "B"}:
                n_high_tier += 1
            if getattr(item, "quote", None):
                n_with_quote += 1

    n_claims = len(claims)

    return {
        "log_evidence": math.log1p(total_items),
        "unlabeled_ratio": 1.0 - (n_with_stance / max(total_items, 1)),
        "low_tier_ratio": 1.0 - (n_high_tier / max(total_items, 1)),
        "log_claims": math.log1p(n_claims),
        "quote_sparsity": 1.0 - (n_with_quote / max(total_items, 1)),
        "n_evidence": float(total_items),
        "n_claims": float(n_claims),
    }
