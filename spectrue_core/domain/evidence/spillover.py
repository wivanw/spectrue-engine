"""Evidence spillover routing within claim clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


NormalizeUrl = Callable[[str], str]
SlotsFromAssertionKey = Callable[[str], set[str]]
RequiredSlotsForTarget = Callable[[str], set[str]]
MergeCovers = Callable[[Any, set[str]], set[str]]
ClaimEventSignature = Callable[[dict[str, Any]], Any]
EvidenceEventSignature = Callable[[dict[str, Any]], Any]
SignatureCompatible = Callable[[Any, Any], bool]


@dataclass(frozen=True)
class SpilloverChoice:
    claim_id: str
    cluster_id: str
    count: int
    urls: list[str]
    topic_boost_used: int


@dataclass(frozen=True)
class SpilloverResult:
    combined_sources: list[dict[str, Any]]
    evidence_by_claim: dict[str, list[dict[str, Any]]]
    transferred_items: list[dict[str, Any]]
    transferred_total: int
    touched_claims: int
    rejections: dict[str, int]
    choices: list[SpilloverChoice]


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


def _claim_assertion_keys(claim: dict[str, Any]) -> tuple[set[str], set[str]]:
    """
    Return (fact_keys, context_keys) from claim.assertions[] if present.
    Deterministic and schema-driven.
    """
    fact: set[str] = set()
    ctx: set[str] = set()
    assertions = claim.get("assertions")
    if not isinstance(assertions, list):
        return fact, ctx

    for a in assertions:
        if not isinstance(a, dict):
            continue
        key = a.get("key")
        if not key:
            continue
        dim = str(a.get("dimension") or "FACT").upper()
        if dim == "CONTEXT":
            ctx.add(str(key))
        else:
            fact.add(str(key))
    return fact, ctx


def _is_transfer_candidate(src: dict[str, Any]) -> bool:
    """
    Conservative: transfer only evidence with an explainability anchor.
    No text heuristics.
    """
    stance = str(src.get("stance") or "").upper()
    if stance in {"IRRELEVANT"}:
        return False
    # Prevent cascade: transferred items must not be used as donors.
    if src.get("provenance") == "transferred":
        return False
    # Require an anchor for downstream explainability/judge
    if src.get("quote") or src.get("quote_span") or src.get("contradiction_span"):
        return True
    return False


def _compatible_for_claim(src: dict[str, Any], fact_keys: set[str], context_keys: set[str]) -> bool:
    """
    Deterministic compatibility using:
    - assertion_key (evidence -> which assertion it applies to)
    - stance class (SUPPORT/REFUTE vs CONTEXT/MENTION)
    - claim assertions dimension (FACT/CONTEXT)
    """
    akey = str(src.get("assertion_key") or "")
    stance = str(src.get("stance") or "").upper()

    # Legacy whole-claim evidence: allow (routing v2 keeps it conservative elsewhere)
    if not akey:
        return True

    if stance in {"SUPPORT", "REFUTE", "MIXED"}:
        return akey in fact_keys

    if stance in {"CONTEXT", "MENTION"}:
        return akey in context_keys

    return False


def _extract_verification_target(claim: dict[str, Any]) -> str:
    """
    Extract verification_target from claim metadata (schema-driven).
    Handles both flattened and nested representations.
    """
    md = claim.get("metadata")
    if isinstance(md, dict) and md.get("verification_target"):
        return str(md.get("verification_target"))
    if claim.get("verification_target"):
        return str(claim.get("verification_target"))
    return ""


def _covers_ok_for_claim(
    src: dict[str, Any],
    claim: dict[str, Any],
    slots_from_assertion_key: SlotsFromAssertionKey,
    required_slots_for_verification_target: RequiredSlotsForTarget,
    merge_covers: MergeCovers,
) -> bool:
    """
    Deterministic compatibility using slots:
    required_slots(verification_target) must intersect evidence covers.
    """
    vt = _extract_verification_target(claim)
    required = required_slots_for_verification_target(vt)
    if not required:
        return True

    akey = str(src.get("assertion_key") or "")
    covers = src.get("covers")

    # Relaxed: if evidence has no structured metadata, don't reject by slots
    if not akey and not covers:
        return True

    derived = slots_from_assertion_key(akey)
    merged = merge_covers(covers, derived)

    return bool(merged & required)


def _event_ok_for_claim(
    src: dict[str, Any],
    claim: dict[str, Any],
    claim_event_signature: ClaimEventSignature,
    evidence_event_signature: EvidenceEventSignature,
    signature_compatible: SignatureCompatible,
) -> bool:
    """
    Deterministic gate using event signatures.
    """
    c_sig = claim_event_signature(claim)
    e_sig = evidence_event_signature(src)

    # Relaxed: if either has no signature, assume compatible for spillover
    if not c_sig or not e_sig:
        return True

    return signature_compatible(c_sig, e_sig)


def _score_for_transfer(src: dict[str, Any]) -> float:
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


def _claim_topic_signature(claim: dict[str, Any]) -> set[str]:
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


def _topic_overlap_boost(origin_claim: dict[str, Any], target_claim: dict[str, Any]) -> float:
    """
    Soft prior: if origin/target claims share topic signature, boost score slightly.
    No filtering. Purely ranking.
    """
    a = _claim_topic_signature(origin_claim)
    b = _claim_topic_signature(target_claim)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    # Bounded linear boost (deterministic, not a threshold)
    return min(0.10, 0.02 * inter)


def _stable_key(src: dict[str, Any], normalize_url: NormalizeUrl) -> tuple[str, str]:
    """
    Deterministic tie-breaker key.
    We use normalized URL (primary) + domain (secondary).
    """
    url = src.get("url") or ""
    dom = src.get("domain") or ""
    try:
        nurl = normalize_url(str(url)) if url else ""
    except Exception:
        nurl = str(url)
    return (nurl, str(dom))


def compute_spillover(
    sources: list[dict[str, Any]],
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
    evidence_by_claim: dict[str, list[dict[str, Any]]] | None = None,
) -> SpilloverResult:
    if evidence_by_claim is None:
        evidence_by_claim = _group_sources_by_claim([s for s in sources if isinstance(s, dict)])

    # claim_id -> claim dict
    claim_lookup: dict[str, dict[str, Any]] = {}
    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("claim_id") or f"c{idx + 1}")
        claim_lookup[cid] = c

    # cluster_id -> claim_ids
    cluster_to_ids: dict[str, list[str]] = {}
    for cid, clid in cluster_map.items():
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

        fact_keys, context_keys = _claim_assertion_keys(target_claim)

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
                if not _is_transfer_candidate(src):
                    rejections["not_candidate"] += 1
                    continue
                if not _compatible_for_claim(src, fact_keys, context_keys):
                    # Relaxed check: if target claim has no defined assertions, don't reject by key
                    if fact_keys or context_keys:
                        rejections["compat_assertion"] += 1
                        continue
                if not _covers_ok_for_claim(
                    src,
                    target_claim,
                    slots_from_assertion_key,
                    required_slots_for_verification_target,
                    merge_covers,
                ):
                    rejections["required_slots"] += 1
                    continue
                if not _event_ok_for_claim(
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
                base = _score_for_transfer(src)
                boost = _topic_overlap_boost(origin_claim, target_claim) if origin_claim else 0.0
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
                _stable_key(t[2], normalize_url)[0],
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
