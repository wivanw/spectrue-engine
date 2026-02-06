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


def apply_dedup(*, sources: list[dict]) -> dict[str, int]:
    updated = 0
    exact_groups = 0
    sim_groups = 0
    seen_hash = set()
    seen_sim = set()

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

        updated += 1

    return {"items": updated, "exact_groups": exact_groups, "similar_groups": sim_groups}
