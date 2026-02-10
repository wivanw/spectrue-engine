"""Corroboration counters for evidence items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _stance_class(s: dict[str, Any]) -> str:
    return str(s.get("stance") or "").upper()


def _is_precision(s: dict[str, Any]) -> bool:
    # Precision evidence = has direct anchor
    return bool(s.get("quote_span") or s.get("contradiction_span") or s.get("quote"))


@dataclass(frozen=True)
class CorroborationResult:
    by_claim: dict[str, dict[str, Any]]
    avg_precise_support: float
    avg_corr_support: float


def compute_corroboration_by_claim(
    sources: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    evidence_by_claim: dict[str, list[dict[str, Any]]] | None = None,
) -> CorroborationResult:
    if evidence_by_claim is None:
        evidence_by_claim = {}
        for s in sources:
            if not isinstance(s, dict):
                continue
            cid = s.get("claim_id")
            if not cid:
                continue
            evidence_by_claim.setdefault(str(cid), []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for c in claims:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("claim_id") or "")
        if not cid:
            continue
        items = [x for x in evidence_by_claim.get(cid, []) if isinstance(x, dict)]

        pub_support = set()
        pub_refute = set()
        clu_support = set()
        clu_refute = set()
        exact_hashes = set()
        pubs_all = set()

        for s in items:
            st = _stance_class(s)
            pub = str(s.get("publisher_id") or "")
            if pub:
                pubs_all.add(pub)
            ch = str(s.get("content_hash") or "")
            if ch:
                exact_hashes.add(ch)

            # corroboration clusters count any support/refute
            scid = str(s.get("similar_cluster_id") or "")
            if st in {"SUPPORT"} and scid:
                clu_support.add(scid)
            if st in {"REFUTE"} and scid:
                clu_refute.add(scid)

            # precision publishers count only direct anchors
            if _is_precision(s):
                if st in {"SUPPORT"} and pub:
                    pub_support.add(pub)
                if st in {"REFUTE"} and pub:
                    pub_refute.add(pub)

        out[cid] = {
            "precision_publishers_support": len(pub_support),
            "precision_publishers_refute": len(pub_refute),
            "corroboration_clusters_support": len(clu_support),
            "corroboration_clusters_refute": len(clu_refute),
            "unique_publishers_total": len(pubs_all),
            "exact_content_groups": len(exact_hashes),
            "evidence_items": len(items),
        }

    avg_precise = round(
        sum(v["precision_publishers_support"] for v in out.values()) / max(1, len(out)),
        3,
    )
    avg_corr = round(
        sum(v["corroboration_clusters_support"] for v in out.values()) / max(1, len(out)),
        3,
    )

    return CorroborationResult(
        by_claim=out,
        avg_precise_support=avg_precise,
        avg_corr_support=avg_corr,
    )
