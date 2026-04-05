"""Evidence deduplication use cases."""

from __future__ import annotations

from spectrue_core.domain.evidence.deduplication import (
    evidence_text_payload,
    normalize_publisher,
    normalize_text_for_hash,
    sha256_hex,
    simhash64,
    simhash_bucket_id,
)


_TIER_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}


def _source_quality(s: dict) -> float:
    """Score a source for dedup ranking: higher = better."""
    tier = str(s.get("tier") or "D").upper()[:1]
    tier_score = _TIER_RANK.get(tier, 0)
    provider_score = float(s.get("provider_score") or s.get("score") or 0)
    has_quote = 1.0 if s.get("quote") or s.get("quote_span") else 0.0
    return tier_score * 10 + provider_score + has_quote


def apply_dedup(*, sources: list[dict], filter_duplicates: bool = True) -> tuple[list[dict], dict[str, int]]:
    """Annotate sources with dedup fingerprints and optionally remove exact duplicates.

    Returns (filtered_sources, stats).
    """
    exact_groups = 0
    sim_groups = 0
    seen_hash: set[str] = set()
    seen_sim: set[str] = set()

    for s in sources:
        if not isinstance(s, dict):
            continue
        domain = str(s.get("domain") or "")
        pub = normalize_publisher(domain)
        s["publisher_id"] = pub

        payload = evidence_text_payload(s)
        norm = normalize_text_for_hash(payload)
        ch = sha256_hex(norm) if norm else ""
        s["content_hash"] = ch

        sh = simhash64(payload) if payload else 0
        scid = simhash_bucket_id(sh, prefix_bits=16) if payload else ""
        s["similar_cluster_id"] = scid

        if ch and ch not in seen_hash:
            seen_hash.add(ch)
            exact_groups += 1
        if scid and scid not in seen_sim:
            seen_sim.add(scid)
            sim_groups += 1

    if not filter_duplicates:
        return sources, {"items": len(sources), "exact_groups": exact_groups, "similar_groups": sim_groups, "removed": 0}

    # Group by content_hash, keep best source per group
    by_hash: dict[str, list[dict]] = {}
    no_hash: list[dict] = []
    for s in sources:
        ch = s.get("content_hash", "")
        if ch:
            by_hash.setdefault(ch, []).append(s)
        else:
            no_hash.append(s)

    filtered: list[dict] = list(no_hash)
    removed = 0
    for group in by_hash.values():
        best = max(group, key=_source_quality)
        filtered.append(best)
        removed += len(group) - 1

    return filtered, {
        "items": len(sources),
        "kept": len(filtered),
        "removed": removed,
        "exact_groups": exact_groups,
        "similar_groups": sim_groups,
    }
