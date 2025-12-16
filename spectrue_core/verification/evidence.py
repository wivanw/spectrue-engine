from spectrue_core.verification.evidence_pack import (
    ArticleContext, Claim, ClaimMetrics, ConfidenceConstraints,
    EvidenceMetrics, EvidencePack, SearchResult, compute_confidence_cap,
)
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trace import Trace

def build_evidence_pack(
    *,
    fact: str,
    claims: list[Claim] | None,
    sources: list[dict],
    search_results_clustered: list[SearchResult] | None = None,
    article_context: ArticleContext | None = None,
    content_lang: str | None = None,
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
                type="core",
                importance=1.0,
                evidence_requirement={
                    "needs_primary_source": False,
                    "needs_independent_2x": True,
                },
                search_queries=[],
            )
        ]
    
    # 3. Convert sources to SearchResult format with enrichments
    search_results: list[SearchResult] = []
        
    if search_results_clustered:
        search_results = search_results_clustered
        seen_domains = set()
        for r in search_results:
            d = r.get("domain") or get_registrable_domain(r.get("url") or "")
            if d:
                r["domain"] = d
                seen_domains.add(d)
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
                claim_id="c1",
                url=url,
                domain=domain,
                title=s.get("title", ""),
                snippet=s.get("snippet", ""),
                content_excerpt=(s.get("content") or s.get("extracted_content") or "")[:1500],
                published_at=s.get("published_date"),
                source_type=source_type,  # type: ignore
                stance=stance,  # type: ignore
                relevance_score=float(s.get("relevance_score", 0.0) or 0.0),
                key_snippet=None,
                quote_matches=[],
                is_trusted=bool(s.get("is_trusted")),
                is_duplicate=is_dup,
                duplicate_of=None,
            )
            search_results.append(search_result)
    
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
    )
    
    # 5. Compute confidence constraints
    cap_per_claim: dict[str, float] = {}
    cap_reasons: list[str] = []
    global_cap = 1.0
    
    for cid, metrics in claim_metrics.items():
        cap, reason = compute_confidence_cap(metrics)
        cap_per_claim[cid] = cap
        cap_reasons.append(f"{cid}: {reason}")
        global_cap = min(global_cap, cap)
    
    constraints = ConfidenceConstraints(
        cap_per_claim=cap_per_claim,
        global_cap=global_cap,
        cap_reasons=cap_reasons,
    )
    
    # 6. Build final Evidence Pack
    pack = EvidencePack(
        article=article_context,
        original_fact=fact,
        claims=claims,
        search_results=search_results,
        metrics=evidence_metrics,
        constraints=constraints,
    )
    
    Trace.event("evidence_pack.built", {
        "total_sources": total_sources,
        "unique_domains": unique_domains,
        "duplicate_ratio": round(duplicate_ratio, 2),
        "global_cap": round(global_cap, 2),
        "cap_reasons": cap_reasons,
    })
    
    return pack
