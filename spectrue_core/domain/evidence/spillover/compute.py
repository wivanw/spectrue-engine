"""
Main spillover computation logic.
"""
from typing import Any
from .models import (
    SpilloverResult, SpilloverChoice,
    NormalizeUrl, SlotsFromAssertionKey, RequiredSlotsForTarget, MergeCovers,
    ClaimEventSignature, EvidenceEventSignature, SignatureCompatible
)
from .criteria import (
    claim_assertion_keys, is_transfer_candidate, compatible_for_claim,
    covers_ok_for_claim, event_ok_for_claim
)
from .ranking import score_for_transfer, topic_overlap_boost, stable_key


def _group_sources_by_claim(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        cid = s.get("claim_id")
        if not cid:
            continue
        grouped.setdefault(str(cid), []).append(s)
    return grouped


def compute_spillover(
    *,
    sources: list[Any],
    claims: list[dict[str, Any]],
    cluster_map: dict[str, str],
    top_k: int,
    normalize_url: NormalizeUrl,
    slots_from_assertion_key: SlotsFromAssertionKey,
    required_slots_for_verification_target: RequiredSlotsForTarget,
    merge_covers: MergeCovers,
    claim_event_signature: ClaimEventSignature,
    evidence_event_signature: EvidenceEventSignature,
    signature_compatible: SignatureCompatible,
    evidence_by_claim: dict | None = None,
) -> SpilloverResult:
    """
    Route evidence between claims in the same cluster.
    """
    if evidence_by_claim is None:
        evidence_by_claim = _group_sources_by_claim([s for s in sources if isinstance(s, dict)])

    claim_lookup = {str(c.get("id")): c for c in claims if c.get("id")}
    cluster_to_ids: dict[str, list[str]] = {}
    for cid, clid in cluster_map.items():
        if cid in claim_lookup:
            cluster_to_ids.setdefault(clid, []).append(cid)

    rejections = {
        "not_candidate": 0,
        "compat_assertion": 0,
        "required_slots": 0,
        "event_signature": 0,
        "dedup": 0,
    }
    transferred_items: list[dict[str, Any]] = []
    transferred_total = 0
    touched_claims = 0
    choices: list[SpilloverChoice] = []

    for target_id, target_claim in claim_lookup.items():
        clid = cluster_map.get(target_id)
        if not clid:
            continue
        peers = cluster_to_ids.get(clid, [])
        if len(peers) <= 1:
            continue

        fact_keys, context_keys = claim_assertion_keys(target_claim)

        # Existing URLs for dedup
        existing_urls: set[str] = set()
        for s in evidence_by_claim.get(target_id, []):
            url = s.get("url")
            if url:
                existing_urls.add(normalize_url(str(url)))

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        topic_boost_used = 0
        for peer_id in peers:
            if peer_id == target_id:
                continue
            for src in evidence_by_claim.get(peer_id, []):
                if not isinstance(src, dict):
                    continue
                if not is_transfer_candidate(src):
                    rejections["not_candidate"] += 1
                    continue
                if not compatible_for_claim(src, fact_keys, context_keys):
                    # Relaxed check: if target claim has no defined assertions, don't reject by key
                    if fact_keys or context_keys:
                        rejections["compat_assertion"] += 1
                        continue
                if not covers_ok_for_claim(
                    src,
                    target_claim,
                    slots_from_assertion_key,
                    required_slots_for_verification_target,
                    merge_covers,
                ):
                    rejections["required_slots"] += 1
                    continue
                if not event_ok_for_claim(
                    src,
                    target_claim,
                    claim_event_signature,
                    evidence_event_signature,
                    signature_compatible,
                ):
                    rejections["event_signature"] += 1
                    continue
                url = src.get("url")
                if url and normalize_url(str(url)) in existing_urls:
                    rejections["dedup"] += 1
                    continue

                origin_claim = claim_lookup.get(peer_id)
                base = score_for_transfer(src)
                boost = topic_overlap_boost(origin_claim, target_claim) if origin_claim else 0.0
                if boost > 0:
                    topic_boost_used += 1
                candidates.append((base + boost, peer_id, src))

        if not candidates:
            continue

        # Stable deterministic ordering:
        # 1) score desc
        # 2) normalized url asc
        # 3) origin claim id asc
        candidates.sort(
            key=lambda t: (
                -t[0],
                stable_key(t[2], normalize_url)[0],
                str(t[1]),
            )
        )
        chosen = candidates[:top_k]
        if chosen:
            touched_claims += 1

        chosen_urls = []
        for _, origin_id, src in chosen:
            url = src.get("url")
            if url:
                existing_urls.add(normalize_url(str(url)))
                chosen_urls.append(normalize_url(str(url)))
            merged = dict(src)
            merged["claim_id"] = target_id
            merged["provenance"] = "transferred"
            merged["origin_claim_id"] = origin_id
            transferred_items.append(merged)
            transferred_total += 1

        choices.append(
            SpilloverChoice(
                claim_id=target_id,
                cluster_id=clid,
                count=len(chosen_urls),
                urls=chosen_urls[:5],
                topic_boost_used=topic_boost_used,
            )
        )

    if not transferred_items:
        return SpilloverResult(
            combined_sources=list(sources),
            evidence_by_claim=_group_sources_by_claim([s for s in sources if isinstance(s, dict)]),
            transferred_items=[],
            transferred_total=0,
            touched_claims=0,
            rejections=rejections,
            choices=choices,
        )

    combined_sources = list(sources) + transferred_items
    by_claim2 = _group_sources_by_claim([s for s in combined_sources if isinstance(s, dict)])

    return SpilloverResult(
        combined_sources=combined_sources,
        evidence_by_claim=by_claim2,
        transferred_items=transferred_items,
        transferred_total=transferred_total,
        touched_claims=touched_claims,
        rejections=rejections,
        choices=choices,
    )
