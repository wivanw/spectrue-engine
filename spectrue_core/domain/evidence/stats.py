"""Evidence stats aggregation for deep v2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


NormalizeUrl = Callable[[str], str]


def _norm_domain(src: dict[str, Any]) -> str:
    d = src.get("domain") or ""
    d = str(d).strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def _has_direct_anchor(src: dict[str, Any]) -> bool:
    return bool(src.get("quote_span") or src.get("contradiction_span") or src.get("quote"))


def _collect_slots(src: dict[str, Any]) -> set[str]:
    slots = set()
    covers = src.get("covers")
    if isinstance(covers, list):
        for x in covers:
            s = str(x).strip().lower()
            if s:
                slots.add(s)
    return slots


def deterministic_explainability(stats: dict[str, Any], bayesian_p: float = 0.5) -> float:
    """
    Deterministic explainability score in [0,1] based on observable artifacts:
    - direct anchors (quote_span/contradiction_span/quote)
    - number of unique domains (diversity)
    - slot coverage (covers[])
    - bayesian sufficiency (proof density)
    This is NOT a verdict; it's a traceability score.
    """
    direct = float(stats.get("direct_anchors", 0))
    uniq = float(stats.get("unique_domains", 0))
    slots = float(stats.get("covered_slots", 0))
    # Saturating functions (no thresholds): 1 - exp(-k*x)
    a1 = 1.0 - math.exp(-0.9 * direct)
    a2 = 1.0 - math.exp(-0.6 * uniq)
    a3 = 1.0 - math.exp(-0.4 * slots)
    a4 = max(0.0, min(1.0, float(bayesian_p)))
    # Weighted blend (fixed, documented)
    # Bayesian consensus now accounts for 30% of the deterministic signal
    out = 0.40 * a1 + 0.15 * a2 + 0.15 * a3 + 0.30 * a4
    return max(0.0, min(1.0, out))


@dataclass
class EvidenceStats:
    """
    Evidence-specific statistics, separated from BudgetState.
    
    Used to track quality signals like direct evidence, coverage slots,
    and distinct domains across the whole verification run.
    """
    sources_observed: int = 0
    sources_with_quote: int = 0
    direct_evidence: int = 0
    unique_domains: set[str] = field(default_factory=set)
    coverage_slots: set[str] = field(default_factory=set)

    def observe(self, ev: Any) -> None:
        """Observe an EvidenceItem or dict and update statistics."""
        self.sources_observed += 1

        # Handle both EvidenceItem objects and dicts
        quote = getattr(ev, "quote", None) if not isinstance(ev, dict) else ev.get("quote")
        if quote:
            self.sources_with_quote += 1

        role = getattr(ev, "evidence_role", "indirect") if not isinstance(ev, dict) else ev.get("evidence_role", "indirect")
        if role == "direct":
            self.direct_evidence += 1

        domain = getattr(ev, "domain", "") if not isinstance(ev, dict) else ev.get("domain", "")
        if domain:
            self.unique_domains.add(domain)

        covers = getattr(ev, "covers", []) if not isinstance(ev, dict) else ev.get("covers", [])
        for c in covers:
            self.coverage_slots.add(c)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a serializable dictionary."""
        return {
            "sources_observed": self.sources_observed,
            "sources_with_quote": self.sources_with_quote,
            "direct_evidence": self.direct_evidence,
            "unique_domain_count": len(self.unique_domains),
            "coverage_slots": list(self.coverage_slots),
        }


def build_evidence_stats(evidence_items: Any) -> Any:
    """
    Build EvidenceStats from a tuple of EvidenceItemFrames.
    """
    from spectrue_core.domain.claims.frame import (
        EvidenceStats as FrameEvidenceStats,
        EvidenceStanceStats,
    )

    total_sources = len(evidence_items)
    support_sources = 0
    refute_sources = 0
    context_sources = 0
    high_trust_sources = 0
    direct_quotes = 0
    unique_publishers = set()
    exact_dupes = 0
    similar_clusters = set()

    # Stance specific tracking
    support_publishers = set()
    refute_publishers = set()
    support_clusters = set()
    refute_clusters = set()

    seen_content_hashes = set()

    for item in evidence_items:
        # Deduplication check
        if hasattr(item, "content_hash") and item.content_hash:
            if item.content_hash in seen_content_hashes:
                exact_dupes += 1
            else:
                seen_content_hashes.add(item.content_hash)
        
        # Publisher tracking
        if hasattr(item, "publisher_id") and item.publisher_id:
            unique_publishers.add(item.publisher_id)

        # Cluster tracking
        if hasattr(item, "similar_cluster_id") and item.similar_cluster_id:
            similar_clusters.add(item.similar_cluster_id)

        # Stance counting
        stance = (getattr(item, "stance", "") or "").upper()
        match stance:
            case "SUPPORT" | "SUP":
                support_sources += 1
                if hasattr(item, "publisher_id") and item.publisher_id:
                    support_publishers.add(item.publisher_id)
                if hasattr(item, "similar_cluster_id") and item.similar_cluster_id:
                    support_clusters.add(item.similar_cluster_id)
            case "REFUTE" | "REF":
                refute_sources += 1
                if hasattr(item, "publisher_id") and item.publisher_id:
                    refute_publishers.add(item.publisher_id)
                if hasattr(item, "similar_cluster_id") and item.similar_cluster_id:
                    refute_clusters.add(item.similar_cluster_id)
            case _:
                context_sources += 1

        # Quality signals
        if getattr(item, "quote", None):
            direct_quotes += 1
        
        tier = (getattr(item, "source_tier", "") or "").upper()
        if tier in ("A", "A'", "A_PRIME"):
            high_trust_sources += 1

    return FrameEvidenceStats(
        total_sources=total_sources,
        support_sources=support_sources,
        refute_sources=refute_sources,
        context_sources=context_sources,
        high_trust_sources=high_trust_sources,
        direct_quotes=direct_quotes,
        conflicting_evidence=(support_sources > 0 and refute_sources > 0),
        missing_sources=(total_sources == 0),
        missing_direct_quotes=(direct_quotes == 0),
        exact_dupes_total=exact_dupes,
        similar_clusters_total=len(similar_clusters),
        publishers_total=len(unique_publishers),
        support=EvidenceStanceStats(
            precision_publishers=len(support_publishers),
            corroboration_clusters=len(support_clusters),
        ),
        refute=EvidenceStanceStats(
            precision_publishers=len(refute_publishers),
            corroboration_clusters=len(refute_clusters),
        ),
    )


@dataclass(frozen=True)
class EvidenceStatsResult:
    by_claim: dict[str, dict[str, Any]]
    avg_sources: float
    avg_explainability: float


def compute_evidence_stats_by_claim(
    sources: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    cluster_map: dict[str, str] | None,
    cluster_sufficiency: dict[str, float] | None,
    normalize_url: NormalizeUrl,
    evidence_by_claim: dict[str, list[dict[str, Any]]] | None = None,
) -> EvidenceStatsResult:
    if evidence_by_claim is None:
        evidence_by_claim = {}
        for s in sources:
            if not isinstance(s, dict):
                continue
            cid = s.get("claim_id")
            if not cid:
                continue
            evidence_by_claim.setdefault(str(cid), []).append(s)

    cluster_map = cluster_map or {}
    cluster_sufficiency = cluster_sufficiency or {}

    stats_by_claim: dict[str, dict[str, Any]] = {}
    for c in claims:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("claim_id") or "")
        if not cid:
            continue
        items = [x for x in evidence_by_claim.get(cid, []) if isinstance(x, dict)]

        unique_urls = set()
        unique_domains = set()
        direct_anchors = 0
        transferred = 0
        best_tier: str | None = None
        slots = set()

        def _rank(t: str | None) -> int:
            return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(str(t).strip().upper(), 0)

        for s in items:
            url = s.get("url")
            if url:
                try:
                    unique_urls.add(normalize_url(str(url)))
                except Exception:
                    unique_urls.add(str(url))
            d = _norm_domain(s)
            if d:
                unique_domains.add(d)
            if _has_direct_anchor(s):
                direct_anchors += 1
            if s.get("provenance") == "transferred":
                transferred += 1
            slots |= _collect_slots(s)

            tier = s.get("tier") or s.get("source_tier")
            if tier and (best_tier is None or _rank(tier) > _rank(best_tier)):
                best_tier = str(tier).strip().upper()

        # Retrieve Bayesian sufficiency from cluster search
        cluster_id = cluster_map.get(cid)
        bayesian_p = cluster_sufficiency.get(cluster_id, 0.5) if cluster_id else 0.5

        stats = {
            "sources_observed": len(items),
            "unique_urls": len(unique_urls),
            "unique_domains": len(unique_domains),
            "direct_anchors": direct_anchors,
            "covered_slots": len(slots),
            "transferred": transferred,
            "bayesian_p": bayesian_p,
            "best_tier": best_tier,
        }
        stats["A_deterministic"] = deterministic_explainability(stats, bayesian_p=bayesian_p)
        stats_by_claim[cid] = stats

    avg_sources = round(
        sum(v["sources_observed"] for v in stats_by_claim.values()) / max(1, len(stats_by_claim)),
        3,
    )
    avg_explainability = round(
        sum(v["A_deterministic"] for v in stats_by_claim.values()) / max(1, len(stats_by_claim)),
        3,
    )

    return EvidenceStatsResult(
        by_claim=stats_by_claim,
        avg_sources=avg_sources,
        avg_explainability=avg_explainability,
    )
