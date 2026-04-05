"""
Scoring and ranking logic for spillover candidates.
"""
from typing import Any
from .models import NormalizeUrl

def score_for_transfer(src: dict[str, Any]) -> float:
    """
    Deterministic ranking using already computed fields.
    Avoids new heuristics or LLM calls.
    """
    try:
        rel = float(src.get("relevance_score", 0.0) or 0.0)
    except Exception:
        rel = 0.0
    stance = str(src.get("stance") or "").upper()
    stance_boost = 0.05 if stance in {"SUPPORT", "REFUTE"} else 0.0
    quote_boost = 0.05 if (src.get("quote") or src.get("quote_span") or src.get("contradiction_span")) else 0.0
    return rel + stance_boost + quote_boost


def claim_topic_signature(claim: dict[str, Any]) -> set[str]:
    """
    Deterministic topic signature for routing priors.
    Uses only structured fields (no text parsing).
    """
    sig: set[str] = set()

    # Legacy category fields
    tg = claim.get("topic_group")
    tk = claim.get("topic_key")
    if tg:
        sig.add(f"topic_group:{str(tg).strip()[:64]}")
    if tk:
        sig.add(f"topic_key:{str(tk).strip()[:64]}")

    # Orchestration metadata topic_tags (if present)
    md = claim.get("metadata")
    if isinstance(md, dict):
        tags = md.get("topic_tags")
        if isinstance(tags, list):
            for t in tags[:16]:
                if t:
                    sig.add(str(t).strip()[:64])

    # Entities / seed terms (bounded)
    se = claim.get("subject_entities")
    if isinstance(se, list):
        for e in se[:5]:
            if e:
                sig.add(f"ent:{str(e).strip()[:64]}")

    st = claim.get("retrieval_seed_terms")
    if isinstance(st, list):
        for s in st[:5]:
            if s:
                sig.add(f"seed:{str(s).strip()[:64]}")

    return sig


def topic_overlap_boost(origin_claim: dict[str, Any], target_claim: dict[str, Any]) -> float:
    """
    Soft prior: if origin/target claims share topic signature, boost score slightly.
    No filtering. Purely ranking.
    """
    a = claim_topic_signature(origin_claim)
    b = claim_topic_signature(target_claim)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    # Bounded linear boost (deterministic, not a threshold)
    return min(0.10, 0.02 * inter)


def stable_key(src: dict[str, Any], normalize_url: NormalizeUrl) -> tuple[str, str]:
    """
    Deterministic tie-breaker key.
    """
    url = src.get("url")
    u_norm = normalize_url(str(url)) if url else ""
    return u_norm, str(src.get("source_id") or "")
