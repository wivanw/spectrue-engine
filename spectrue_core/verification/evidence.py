from spectrue_core.verification.evidence_pack import (
    ArticleContext, Claim, ClaimMetrics, ConfidenceConstraints,
    EvidenceMetrics, EvidencePack, SearchResult, AssertionMetrics
)
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.trusted_sources import get_tier_ceiling_for_domain
from typing import Any
import logging

logger = logging.getLogger(__name__)

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
    
    # 3. Convert sources to SearchResult format with enrichments
    search_results: list[SearchResult] = []
        
    # If clustering ran and returned a list (even empty), use it.
    # Only fallback to raw sources if it returned None (error/timeout).
    if search_results_clustered is not None:
        search_results = search_results_clustered
        logger.info("Using clustered evidence: %d items", len(search_results))
        seen_domains = set()
        for r in search_results:
            d = r.get("domain") or get_registrable_domain(r.get("url") or "")
            if d:
                # FIX 3: Detect duplicates for clustered results too
                is_dup = d in seen_domains
                r["is_duplicate"] = is_dup
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
                t = (r.get("source_type") or "").upper()
                # Rough mapping back to tiers if needed, or rely on evidence_tier if present in SearchResult (it's not by default, purely derived)
                # Actually SearchResult doesn't have evidence_tier, it has source_type.
                # Let's map source_type to tier for metrics
                tier = "C"
                if r.get("is_trusted"): tier = "B"
                if r.get("source_type") == "primary": tier = "A"
                if r.get("source_type") == "official": tier = "A'"
                if r.get("source_type") == "social": tier = "D"
                
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
    
    max_cap_found = 0.35 # Default to Tier D (Social) if no sources
    
    if search_results:
        caps = []
        for r in search_results:
            d = r.get("domain") or get_registrable_domain(r.get("url") or "")
            tier_override = r.get("evidence_tier") 
            
            # Check for explicitly promoted tiers (M67 Stage 3)
            # Tier A' (Verified Official Social) -> 0.75 (same as Trusted Media)
            if tier_override == "A'":
                caps.append(0.75)
            else:
                c = get_tier_ceiling_for_domain(d)
                caps.append(c)
        
        if caps:
            max_cap_found = max(caps)
            
    global_cap = float(max_cap_found)
    cap_reasons = [f"Ceiling determined by strongest source tier (limit {global_cap:.2f})"]
    
    constraints = ConfidenceConstraints(
        cap_per_claim={},
        global_cap=global_cap,
        cap_reasons=cap_reasons,
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # M78.1: Split sources into scored (for verdict) and context (for UX)
    # ─────────────────────────────────────────────────────────────────────────
    SCORING_STANCES = {"support", "contradict", "refute", "mixed", "neutral"}
    CONTEXT_STANCES = {"context", "irrelevant", "mention", "unclear"}
    
    scored_sources: list[SearchResult] = []
    context_sources: list[SearchResult] = []
    
    for r in search_results:
        stance = (r.get("stance") or "unclear").lower()
        if stance in SCORING_STANCES:
            scored_sources.append(r)
        else:
            context_sources.append(r)
    
    # 6. Build final Evidence Pack
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
    
    return pack

