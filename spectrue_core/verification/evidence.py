from spectrue_core.verification.evidence_pack import (
    ArticleContext, Claim, ClaimMetrics, ConfidenceConstraints,
    EvidenceItem, EvidenceMetrics, EvidencePack, EvidencePackStats,
    SearchResult, AssertionMetrics
)
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trace import Trace
from typing import Any
import re
import logging

logger = logging.getLogger(__name__)

HIGH_TIER_SET = {"A", "A'", "B"}
LOW_TIER_SET = {"C", "D"}
TIER_RANK = {"D": 1, "C": 2, "B": 3, "A'": 4, "A": 4}


def _normalize_tier(*, tier_raw: str | None, source_type: str | None) -> str:
    if tier_raw:
        return str(tier_raw).strip().upper()
    stype = (source_type or "").strip().lower()
    if stype == "primary":
        return "A"
    if stype == "official":
        return "A'"
    if stype == "independent_media":
        return "B"
    if stype == "social":
        return "D"
    return "C"


def _tier_rank(tier: str) -> int:
    return TIER_RANK.get(tier, 0)


def is_strong_tier(tier: str | None) -> bool:
    if not tier:
        return False
    return tier in HIGH_TIER_SET


def strongest_tiers_by_claim(
    scored_sources: list[SearchResult],
) -> dict[str, dict[str, str | int | None]]:
    """
    Determine strongest support/refute tiers per claim from scored sources.
    """
    by_claim: dict[str, dict[str, str | int | None]] = {}

    for src in scored_sources:
        if not isinstance(src, dict):
            continue
        claim_id = str(src.get("claim_id") or "").strip()
        if not claim_id:
            claim_id = "c1"

        stance = (src.get("stance") or "").lower()
        if stance not in ("support", "refute", "contradict"):
            continue

        tier = _normalize_tier(
            tier_raw=src.get("evidence_tier"),
            source_type=src.get("source_type"),
        )
        rank = _tier_rank(tier)
        entry = by_claim.setdefault(
            claim_id,
            {
                "support_tier": None,
                "support_rank": 0,
                "refute_tier": None,
                "refute_rank": 0,
            },
        )

        if stance == "support":
            if rank > int(entry["support_rank"] or 0):
                entry["support_rank"] = rank
                entry["support_tier"] = tier
        else:
            if rank > int(entry["refute_rank"] or 0):
                entry["refute_rank"] = rank
                entry["refute_tier"] = tier

    return by_claim


def merge_stance_passes(
    *,
    support_results: list[SearchResult],
    refute_results: list[SearchResult],
    original_sources: list[dict],
) -> list[SearchResult]:
    if not support_results:
        return refute_results
    if not refute_results:
        return support_results
    if len(support_results) != len(refute_results):
        logger.warning(
            "[StanceMerge] Pass count mismatch: support=%d refute=%d",
            len(support_results),
            len(refute_results),
        )
        return support_results

    merged: list[SearchResult] = []
    for idx, support in enumerate(support_results):
        refute = refute_results[idx]
        src = original_sources[idx] if idx < len(original_sources) else {}

        support_quote = support.get("quote_span") or support.get("key_snippet")
        refute_quote = refute.get("contradiction_span") or refute.get("key_snippet")

        support_tier = _normalize_tier(
            tier_raw=support.get("evidence_tier") or src.get("evidence_tier"),
            source_type=support.get("source_type") or src.get("source_type"),
        )
        refute_tier = _normalize_tier(
            tier_raw=refute.get("evidence_tier") or src.get("evidence_tier"),
            source_type=refute.get("source_type") or src.get("source_type"),
        )

        support_rank = _tier_rank(support_tier)
        refute_rank = _tier_rank(refute_tier)

        support_has = (support.get("stance") == "support") and bool(support_quote)
        refute_has = (refute.get("stance") == "refute") and bool(refute_quote)

        if refute_has and refute_rank >= support_rank:
            merged_result = dict(refute)
            merged_result["stance"] = "refute"
            merged_result["pass_type"] = "REFUTE_ONLY"
            merged_result["contradiction_span"] = refute_quote
            merged_result["evidence_tier"] = refute_tier
        elif support_has:
            merged_result = dict(support)
            merged_result["stance"] = "support"
            merged_result["pass_type"] = "SUPPORT_ONLY"
            merged_result["quote_span"] = support_quote
            merged_result["evidence_tier"] = support_tier
            if support_tier in LOW_TIER_SET:
                merged_result["stance_confidence"] = "low"
        elif refute_has:
            merged_result = dict(refute)
            merged_result["stance"] = "refute"
            merged_result["pass_type"] = "REFUTE_ONLY"
            merged_result["contradiction_span"] = refute_quote
            merged_result["evidence_tier"] = refute_tier
        else:
            merged_result = dict(support)
            merged_result["stance"] = "context"
            merged_result["pass_type"] = "SUPPORT_ONLY"

        if merged_result.get("stance") not in ("support", "refute"):
            merged_result["quote_span"] = None
            merged_result["contradiction_span"] = None

        url = merged_result.get("url") or src.get("url") or src.get("link")
        if url:
            merged_result["evidence_refs"] = [url]

        merged.append(merged_result)  # type: ignore

    return merged

def needs_evidence_acquisition_ladder(sources: list[dict]) -> bool:
    """
    Determine whether to escalate via the Evidence Acquisition Ladder (EAL).
    
    M104: EAL is needed if any source lacks a `quote` field.
    Content/snippet alone is not sufficient for evidence scoring.
    """
    if not sources:
        return False
    
    # M104: EAL needed if ANY source lacks quote
    any_missing_quote = any(
        isinstance(s, dict) and not s.get("quote")
        for s in sources
    )
    return any_missing_quote


def extract_quote_candidates(content: str, max_quotes: int = 2) -> list[str]:
    """
    Extract lightweight quote candidates from content.
    """
    if not content:
        return []
    # Split on sentence boundaries.
    parts = [p.strip() for p in re.split(r"[.!?]\s+", content) if p.strip()]
    quotes: list[str] = []
    for part in parts:
        if len(part) < 40:
            continue
        quotes.append(part[:280])
        if len(quotes) >= max_quotes:
            break
    return quotes

def build_evidence_pack(
    *,
    fact: str,
    claims: list[Claim] | None,
    sources: list[dict],
    search_results_clustered: list[SearchResult] | None = None,
    article_context: ArticleContext | None = None,
    content_lang: str | None = None,
    claim_units: list[Any] | None = None,  # M70: Structured ClaimUnits
) -> EvidencePack:
    """
    Build structured Evidence Pack for LLM scorer.
    
    Code computes all metrics and caps; LLM produces verdict within caps.
    This is the contract between code (evidence collector) and LLM (judge).
    """
    
    # 1. Article context (if URL was provided)
    if article_context is None:
        article_context = ArticleContext(
            text_excerpt=fact[:500] if fact else "",
            content_lang=content_lang,
        )
    
    # 2. Claims (if not provided, create single "core" claim)
    if not claims:
        claims = [
            Claim(
                id="c1",
                text=fact,
                normalized_text=fact,  # M62: Use fact as-is
                type="core",
                topic_group="Other",   # M62: Default topic
                importance=1.0,
                check_worthiness=0.5,  # M62: Default worthiness
                evidence_requirement={
                    "needs_primary_source": False,
                    "needs_independent_2x": True,
                },
                search_queries=[],
            )
        ]

    time_sensitive_by_claim: dict[str, bool] = {}
    for claim in claims:
        claim_id = claim.get("id", "c1")
        metadata = claim.get("metadata")
        time_sensitive = bool(getattr(metadata, "time_sensitive", False)) if metadata else False
        req = claim.get("evidence_requirement") or {}
        if isinstance(req, dict) and (req.get("is_time_sensitive") or req.get("needs_recent_source")):
            time_sensitive = True
        time_sensitive_by_claim[claim_id] = time_sensitive

    # 3. Convert sources to SearchResult format with enrichments
    search_results: list[SearchResult] = []
        
    # If clustering ran and returned a list (even empty), use it.
    # Only fallback to raw sources if it returned None (error/timeout).
    default_claim_id = claims[0].get("id") if claims else "c1"
    if search_results_clustered is not None:
        search_results = search_results_clustered
        logger.debug("Using clustered evidence: %d items", len(search_results))
        seen_domains = set()
        for r in search_results:
            d = r.get("domain") or get_registrable_domain(r.get("url") or "")
            if d:
                # FIX 3: Detect duplicates for clustered results too
                is_dup = d in seen_domains
                r["is_duplicate"] = is_dup
                r["domain"] = d
                seen_domains.add(d)
            if not r.get("source_type") or r.get("source_type") == "unknown":
                if r.get("is_primary"):
                    r["source_type"] = "primary"
                elif r.get("is_trusted"):
                    r["source_type"] = "independent_media"
            if not r.get("claim_id"):
                r["claim_id"] = default_claim_id
                Trace.event(
                    "evidence.claim_id.missing",
                    {
                        "assigned_claim_id": default_claim_id,
                        "source_url": r.get("url") or r.get("link"),
                    },
                )
    else:
        seen_domains = set()
        for s in (sources or []):
            url = (s.get("link") or s.get("url") or "").strip()
            domain = get_registrable_domain(url)
            
            # Detect duplicates by domain
            is_dup = domain in seen_domains if domain else False
            if domain:
                seen_domains.add(domain)
            
            # Map source_type based on is_trusted and evidence_tier
            tier = (s.get("evidence_tier") or "").strip().upper()
            is_trusted = bool(s.get("is_trusted"))
            
            if tier == "A":
                source_type = "primary"
            elif tier == "A'":
                source_type = "official"
            elif tier == "B" or is_trusted:
                # Trusted domains are independent media
                source_type = "independent_media"
            elif tier == "D":
                source_type = "social"
            else:
                # Default to aggregator for unknown/C tier
                source_type = "aggregator"
            
            stance = s.get("stance", "unclear")
            
            # Build SearchResult
            search_result = SearchResult(
                claim_id=default_claim_id,
                url=url,
                domain=domain,
                title=s.get("title", ""),
                snippet=s.get("snippet", ""),
                content_excerpt=(s.get("content") or s.get("extracted_content") or "")[:1500],
                published_at=s.get("published_date"),
                source_type=source_type,  # type: ignore
                stance=stance,  # type: ignore
                relevance_score=float(s.get("relevance_score", 0.0) or 0.0),
                timeliness_status=s.get("timeliness_status"),
                key_snippet=None,
                quote_matches=[],
                is_trusted=bool(s.get("is_trusted")),
                is_duplicate=is_dup,
                duplicate_of=None,
            )
            search_results.append(search_result)

    for r in search_results:
        claim_id = r.get("claim_id", "c1")
        if not time_sensitive_by_claim.get(claim_id):
            continue
        status = str(r.get("timeliness_status") or "").lower()
        stance = str(r.get("stance") or "").lower()
        if status == "outdated" and stance == "support":
            r["stance"] = "context"
            r["timeliness_excluded"] = True

    # Enforce quote requirement for SUPPORT/REFUTE
    for r in search_results:
        stance = str(r.get("stance") or "").lower()
        quote = r.get("quote_span") or r.get("contradiction_span") or r.get("key_snippet")
        if stance in ("support", "refute", "contradict") and not quote:
            r["stance"] = "context"
            r["quote_missing"] = True

    seen_domains = set()
    for r in search_results:
        d = r.get("domain") or get_registrable_domain(r.get("url") or "")
        if d:
            is_dup = d in seen_domains
            r["is_duplicate"] = is_dup
            r["domain"] = d
            if not is_dup:
                seen_domains.add(d)

    # 4. Compute metrics
    unique_domains = len(seen_domains)
    total_sources = len(search_results)
    
    duplicate_count = sum(1 for r in search_results if r.get("is_duplicate"))
    duplicate_ratio = duplicate_count / total_sources if total_sources > 0 else 0.0
    
    # Count source types
    type_dist: dict[str, int] = {}
    for r in search_results:
        st = r.get("source_type", "unknown")
        type_dist[st] = type_dist.get(st, 0) + 1
    
    # Per-claim metrics
    claim_metrics: dict[str, ClaimMetrics] = {}
    for claim in claims:
        cid = claim.get("id", "c1")
        claim_sources = [r for r in search_results if r.get("claim_id") == cid]
        
        # Count independent domains for this claim
        claim_domains = {r.get("domain") for r in claim_sources if r.get("domain")}
        independent = len(claim_domains)
        
        # Check for primary/official sources
        primary = any(r.get("source_type") == "primary" for r in claim_sources)
        official = any(r.get("source_type") == "official" for r in claim_sources)
        
        # Stance distribution
        stance_dist: dict[str, int] = {}
        for r in claim_sources:
            st = r.get("stance", "unclear")
            stance_dist[st] = stance_dist.get(st, 0) + 1
        
        # Coverage: ratio of sources with relevance > 0.5
        relevant = sum(1 for r in claim_sources if (r.get("relevance_score") or 0) > 0.5)
        coverage = relevant / len(claim_sources) if claim_sources else 0.0
        
        claim_metrics[cid] = ClaimMetrics(
            independent_domains=independent,
            primary_present=primary,
            official_present=official,
            stance_distribution=stance_dist,
            coverage=coverage,
            freshness_days_median=None,
            source_type_distribution=type_dist,  # type: ignore
            # M62: Include claim metadata for scoring caps
            topic_group=claim.get("topic_group"),
            claim_type=claim.get("type"),
        )
    
    # M70: Per-Assertion Metrics
    assertion_metrics: dict[str, AssertionMetrics] = {}
    if claim_units:
        # Map all assertions
        all_assertions = []
        for cu in claim_units:
            all_assertions.extend(cu.assertions)
            
        for assertion in all_assertions:
            akey = assertion.key
            # Filter evidence for this assertion
            a_evidence = [
                r for r in search_results 
                if r.get("assertion_key") == akey
            ]
            
            support = sum(1 for r in a_evidence if r.get("stance") == "SUPPORT")
            refute = sum(1 for r in a_evidence if r.get("stance") == "REFUTE")
            unavailable = sum(1 for r in a_evidence if r.get("content_status") == "unavailable")
            
            # Tiers present
            tiers = {}
            for r in a_evidence:
                # Map source_type to tier for metrics
                tier = "C"
                if r.get("is_trusted"):
                    tier = "B"
                if r.get("source_type") == "primary":
                    tier = "A"
                if r.get("source_type") == "official":
                    tier = "A'"
                if r.get("source_type") == "social":
                    tier = "D"
                
                tiers[tier] = tiers.get(tier, 0) + 1
                
            assertion_metrics[akey] = AssertionMetrics(
                support_count=support,
                refute_count=refute,
                tier_coverage=tiers,
                primary_present=any(r.get("source_type") == "primary" for r in a_evidence),
                official_present=any(r.get("source_type") == "official" for r in a_evidence),
                content_unavailable_count=unavailable,
            )
    
    # Overall coverage
    coverages = [m.get("coverage", 0) for m in claim_metrics.values()]
    overall_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    
    evidence_metrics = EvidenceMetrics(
        total_sources=total_sources,
        unique_domains=unique_domains,
        duplicate_ratio=duplicate_ratio,
        per_claim=claim_metrics,
        overall_coverage=overall_coverage,
        freshness_days_median=None,
        source_type_distribution=type_dist,
        per_assertion=assertion_metrics, # M70
    )
    
    # 5. Initialize confidence constraints (Code is Law: M67)
    # The Code defines the Maximum Possible Probability (Ceiling) a source can contribute.
    # An LLM cannot "hallucinate" high confidence if the Evidence Tier is low.
    
    # Default with no sources: do not cap (tests expect 1.0).
    # M50: Cap logic removed. LLM has full discretion over confidence scores.
    global_cap = 1.0
    cap_reasons = ["Confidence caps disabled (M50)"]
    
    constraints = ConfidenceConstraints(
        cap_per_claim={},
        global_cap=global_cap,
        cap_reasons=cap_reasons,
    )
    
    # M78.1: Split sources into scored (for verdict) and context (for UX)
    # ─────────────────────────────────────────────────────────────────────────
    SCORING_STANCES = {"support", "contradict", "refute", "mixed", "neutral"}
    
    scored_sources: list[SearchResult] = []
    context_sources: list[SearchResult] = []
    
    for r in search_results:
        stance = (r.get("stance") or "unclear").lower()
        if stance in SCORING_STANCES:
            scored_sources.append(r)
        else:
            context_sources.append(r)

    stance_failure = bool(search_results) and not scored_sources
    evidence_metrics["stance_failure"] = stance_failure
    
    # 6. Build canonical evidence items for deterministic scoring
    evidence_items: list[EvidenceItem] = []
    tiers_present: dict[str, int] = {}
    support_count = 0
    refute_count = 0
    context_count = 0
    outdated_count = 0
    item_domains: set[str] = set()
    per_claim_stats: dict[str, dict[str, int]] = {}

    for r in search_results:
        tier = _normalize_tier(
            tier_raw=r.get("evidence_tier"),
            source_type=r.get("source_type"),
        )
        stance_raw = str(r.get("stance") or "").lower()
        if stance_raw in ("support",):
            stance = "SUPPORT"
        elif stance_raw in ("refute", "contradict"):
            stance = "REFUTE"
        elif stance_raw in ("irrelevant",):
            stance = "IRRELEVANT"
        else:
            stance = "CONTEXT"

        quote = r.get("quote_span") or r.get("contradiction_span") or r.get("key_snippet")
        if stance in ("SUPPORT", "REFUTE") and not quote:
            stance = "CONTEXT"

        temporal_flag = str(r.get("timeliness_status") or "unknown").lower()
        if temporal_flag not in ("in_window", "outdated", "unknown"):
            temporal_flag = "unknown"

        source_type = str(r.get("source_type") or "").lower()
        if source_type in ("primary", "official"):
            channel = "authoritative"
        elif source_type in ("independent_media",):
            channel = "reputable_news"
        elif source_type in ("social",):
            channel = "social"
        elif source_type in ("aggregator",):
            channel = "local_media"
        else:
            channel = "low_reliability"

        domain = r.get("domain") or get_registrable_domain(r.get("url") or "")
        if domain:
            item_domains.add(domain)

        evidence_items.append(EvidenceItem(
            url=r.get("url") or "",
            domain=domain or "",
            title=r.get("title"),
            snippet=r.get("snippet"),
            channel=channel,  # type: ignore[arg-type]
            tier=tier,  # type: ignore[arg-type]
            tier_reason=None,
            claim_id=r.get("claim_id") or "c1",
            stance=stance,  # type: ignore[arg-type]
            quote=quote,
            relevance=float(r.get("relevance_score", 0.0) or 0.0),
            published_at=r.get("published_at"),
            temporal_flag=temporal_flag,  # type: ignore[arg-type]
            fetched=r.get("content_status") == "available",
            raw_text_chars=len(r.get("content_excerpt") or ""),
        ))

        claim_id = r.get("claim_id") or "c1"
        claim_stats = per_claim_stats.setdefault(
            str(claim_id),
            {"support": 0, "refute": 0, "context": 0, "with_quote": 0},
        )
        if stance == "SUPPORT":
            claim_stats["support"] += 1
        elif stance == "REFUTE":
            claim_stats["refute"] += 1
        else:
            claim_stats["context"] += 1
        if quote:
            claim_stats["with_quote"] += 1

        tiers_present[tier] = tiers_present.get(tier, 0) + 1
        if stance == "SUPPORT":
            support_count += 1
        elif stance == "REFUTE":
            refute_count += 1
        elif stance == "CONTEXT":
            context_count += 1
        if temporal_flag == "outdated":
            outdated_count += 1

    stats = EvidencePackStats(
        domain_diversity=len(item_domains),
        tiers_present=tiers_present,
        support_count=support_count,
        refute_count=refute_count,
        context_count=context_count,
        outdated_ratio=(outdated_count / len(evidence_items)) if evidence_items else 0.0,
    )

    # 7. Build final Evidence Pack
    pack = EvidencePack(
        article=article_context,
        original_fact=fact,
        claims=claims,
        claim_units=claim_units, # M70
        search_results=search_results,  # All sources (backward compat)
        scored_sources=scored_sources,  # M78.1: For verdict computation
        context_sources=context_sources, # M78.1: For transparency/UX
        metrics=evidence_metrics,
        constraints=constraints,
        claim_id=(claims[0].get("id") if claims else "c1"),
        items=evidence_items,
        stats=stats,
        global_cap=global_cap,
        cap_reasons=cap_reasons,
    )
    
    Trace.event("evidence_pack.built", {
        "total_sources": total_sources,
        "scored_sources": len(scored_sources),
        "context_sources": len(context_sources),
        "unique_domains": unique_domains,
        "duplicate_ratio": round(duplicate_ratio, 2),
        "global_cap": round(global_cap, 2),
        "cap_reasons": cap_reasons,
    })

    Trace.event(
        "evidence.items.summary",
        {
            "per_claim": per_claim_stats,
            "support_total": support_count,
            "refute_total": refute_count,
            "context_total": context_count,
        },
    )
    
    return pack
