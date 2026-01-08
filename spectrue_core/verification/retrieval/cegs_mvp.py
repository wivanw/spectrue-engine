import asyncio
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Set, Tuple
import hashlib

from pydantic import BaseModel, Field

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.coverage_anchors import Anchor, AnchorKind
from spectrue_core.verification.claims.coverage_anchors import extract_all_anchors
from spectrue_core.verification.search.search_escalation import (
    get_escalation_ladder,
    build_query_variants,
    select_topic_from_claim
)

logger = logging.getLogger(__name__)

# Module wiring verification (D) - emitted once at import time
Trace.event("retrieval.cegs.module", {
    "module_path": __name__,
    "file": __file__,
})

# --- Helpers ---

def _normalize_text(text: str) -> Set[str]:
    """Normalize text to set of tokens for overlap checking."""
    return set(re.findall(r"\w{3,}", text.lower()))

def _calculate_jaccard(text1: str, text2: str) -> float:
    set1 = _normalize_text(text1)
    set2 = _normalize_text(text2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def _compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _extract_entities_from_claim(claim: Dict[str, Any]) -> List[str]:
    """Safely extract entities from a claim dict."""
    entities = []
    
    # context_entities
    ce = claim.get("context_entities")
    if isinstance(ce, list):
        entities.extend([str(e) for e in ce if isinstance(e, str)])
        
    # subject_entities - assuming it's a list or similar
    se = claim.get("subject_entities")
    if isinstance(se, list):
        entities.extend([str(e) for e in se if isinstance(e, str)])
        
    # subject
    subj = claim.get("subject")
    if isinstance(subj, str) and subj:
        entities.append(subj)
        
    return entities


def collect_document_entities(claims: List[Dict[str, Any]], max_count: int = 8) -> List[str]:
    """
    Collect top N document-level entities from claims by frequency.
    
    Rules (B):
    - Gather from context_entities and subject_entities
    - Filter: strings with len >= 2
    - Normalize for counting (lowercase/strip)
    - Return top N in original casing (first-seen)
    - Works even if some claims miss fields
    """
    # Track frequency with normalized key, preserve first-seen casing
    freq: Counter = Counter()
    original_casing: Dict[str, str] = {}  # normalized -> first-seen original
    
    for claim in claims:
        entities = _extract_entities_from_claim(claim)
        for entity in entities:
            if not isinstance(entity, str) or len(entity) < 2:
                continue
            normalized = entity.lower().strip()
            if not normalized:
                continue
            freq[normalized] += 1
            if normalized not in original_casing:
                original_casing[normalized] = entity.strip()
    
    # Get top N by frequency, return in original casing
    top_normalized = [key for key, _ in freq.most_common(max_count)]
    return [original_casing[n] for n in top_normalized]


def collect_seed_terms(claims: List[Dict[str, Any]], max_count: int = 4) -> List[str]:
    """
    Collect top N seed terms from claims by frequency.
    
    Used for Q3 query generation (C).
    """
    freq: Counter = Counter()
    original_casing: Dict[str, str] = {}
    
    for claim in claims:
        seed_terms = claim.get("retrieval_seed_terms", [])
        if not isinstance(seed_terms, list):
            continue
        for term in seed_terms:
            if not isinstance(term, str) or len(term) < 2:
                continue
            normalized = term.lower().strip()
            if not normalized:
                continue
            freq[normalized] += 1
            if normalized not in original_casing:
                original_casing[normalized] = term.strip()
    
    top_normalized = [key for key, _ in freq.most_common(max_count)]
    return [original_casing[n] for n in top_normalized]


# --- Core Evidence Models ---

class EvidenceSourceMeta(BaseModel):
    """Metadata about a source found during search."""
    url: str
    title: str
    snippet: str
    score: float
    provider_meta: Dict[str, Any] = Field(default_factory=dict)

class EvidenceItem(BaseModel):
    """A unit of evidence content extracted from a source."""
    url: str
    extracted_text: str
    citations: List[str] = Field(default_factory=list)
    content_hash: str
    source_meta: EvidenceSourceMeta

class EvidenceBundle(BaseModel):
    """Result of matching pool evidence to a specific claim."""
    claim_id: str
    matched_items: List[EvidenceItem]
    cluster_ids: List[str] = Field(default_factory=list)
    coverage_flags: Dict[str, Any] = Field(default_factory=dict)

class EvidencePool(BaseModel):
    """Shared collection of evidence for a verification run."""
    items: List[EvidenceItem] = Field(default_factory=list)
    meta: List[EvidenceSourceMeta] = Field(default_factory=list)

    def add_items(self, new_items: List[EvidenceItem]):
        """Adds new items to the pool, updating meta as needed."""
        self.items.extend(new_items)
        # Add meta for new items if not already present
        existing_urls = {m.url for m in self.meta}
        for item in new_items:
            if item.source_meta.url not in existing_urls:
                self.meta.append(item.source_meta)
                existing_urls.add(item.source_meta.url)

    def match_claim(self, claim: Any) -> EvidenceBundle:
        """
        Delegates to match_claim_to_pool.
        """
        return match_claim_to_pool(claim, self)

# --- Deficit Model ---

class Deficit(BaseModel):
    """Assessment of whether evidence is sufficient for a claim."""
    is_deficit: bool
    reason_codes: List[str] = Field(default_factory=list)
    severity: float = 0.0

# --- T007: Pool Matching ---

def match_claim_to_pool(claim: Dict[str, Any], pool: EvidencePool) -> EvidenceBundle:
    """
    Matches pool evidence to a claim using structural overlap and anchor constraints.
    """
    claim_id = str(claim.get("id") or claim.get("claim_id") or "unknown")
    
    # 1. Prepare Claim Signals
    # Entities
    claim_entities = _normalize_text(" ".join(_extract_entities_from_claim(claim)))
    
    # Anchors
    claim_text = claim.get("normalized_text") or claim.get("text") or ""
    claim_anchors = extract_all_anchors(claim_text)
    
    has_time_anchor = any(a.kind == AnchorKind.TIME for a in claim_anchors)
    has_number_anchor = any(a.kind == AnchorKind.NUMBER for a in claim_anchors)
    
    scored_items: List[Tuple[float, EvidenceItem]] = []
    
    for item in pool.items:
        # Check Content
        content_text = f"{item.source_meta.title} {item.source_meta.snippet} {item.extracted_text}"
        
        # Anchor Constraints (Hard Filters or Strong Penalties?)
        # Spec: "if claim has time anchor -> evidence must contain a time-like token"
        # We will use penalty for MVP or hard filter. Spec says "must contain".
        # Let's check constraints.
        
        passes_constraints = True
        item_anchors = extract_all_anchors(content_text)
        
        if has_time_anchor:
            if not any(a.kind == AnchorKind.TIME for a in item_anchors):
                passes_constraints = False
        
        if has_number_anchor:
             if not any(a.kind == AnchorKind.NUMBER for a in item_anchors):
                 passes_constraints = False
                 
        if not passes_constraints:
            continue
            
        # Entity Overlap Score
        content_tokens = _normalize_text(content_text)
        overlap = len(claim_entities & content_tokens)
        
        # Score = overlap count + small bonus for source score
        score = overlap + (item.source_meta.score * 0.1)
        
        if overlap > 0: # Only consider items with at least some overlap
             scored_items.append((score, item))
             
    # Pick Top N=3
    scored_items.sort(key=lambda x: x[0], reverse=True)
    top_items = [x[1] for x in scored_items[:3]]
    top_scores = [x[0] for x in scored_items[:3]]
    
    bundle = EvidenceBundle(
        claim_id=claim_id,
        matched_items=top_items,
        cluster_ids=[], # TODO: Propagate cluster IDs from retrieval?
        coverage_flags={
            "has_time_anchor": has_time_anchor,
            "has_number_anchor": has_number_anchor,
            "matched_count": len(top_items)
        }
    )
    
    Trace.event("retrieval.pool.match", {
        "claim_id": claim_id,
        "matched_count": len(top_items),
        "top_match_scores": top_scores,
        "reason_codes": ["no_pool_matches"] if not top_items else []
    })
    
    return bundle

# --- T005: Query Planning ---

def build_doc_query_plan(claims: List[Dict[str, Any]], anchors: List[Anchor]) -> List[str]:
    """
    Generates 2-4 document-level queries based on entities and anchors.
    
    Rules (C - Deterministic and Bounded):
    - Q1: top 3 entities + optional date anchor
    - Q2: top 3 entities + optional numeric anchor
    - Q3: top 2 entities + top 2 seed terms (optional)
    - Q4: guideline token for high-risk claims (optional)
    - Each query <= 80 chars
    - Returns 2-4 queries when entities exist, [] otherwise
    """
    # Use improved entity collection (B)
    top_entities = collect_document_entities(claims, max_count=8)
    seed_terms = collect_seed_terms(claims, max_count=4)
    
    # Check for empty entities (B)
    if not top_entities:
        Trace.event("retrieval.doc_plan.no_entities", {
            "claims_count": len(claims),
            "anchor_count": len(anchors),
        })
        # Return empty - caller handles fallback
        Trace.event("retrieval.doc_plan", {
            "queries": [],
            "entity_count_used": 0,
            "anchor_count_used": 0,
            "entity_top": [],
        })
        return []
    
    # Select strongest anchors (TIME > NUMBER)
    time_anchors = [a for a in anchors if a.kind == AnchorKind.TIME]
    number_anchors = [a for a in anchors if a.kind == AnchorKind.NUMBER]
    
    date_anchor = time_anchors[0].span_text if time_anchors else None
    numeric_anchor = number_anchors[0].span_text if number_anchors else None
    
    queries: List[str] = []
    
    # Q1: Top 3 entities + optional date anchor (C)
    q1_parts = top_entities[:3]
    if date_anchor:
        q1_parts = q1_parts + [date_anchor]
    q1 = " ".join(q1_parts)
    queries.append(q1)
    
    # Q2: Top 3 entities + optional numeric anchor (C)
    # Use different entity slice if possible for diversity
    q2_entities = top_entities[:3] if len(top_entities) <= 3 else top_entities[1:4]
    q2_parts = q2_entities[:]
    if numeric_anchor:
        q2_parts = q2_parts + [numeric_anchor]
    elif date_anchor and not numeric_anchor:
        # If no numeric but we have date, use entities only for Q2
        pass
    q2 = " ".join(q2_parts)
    if q2.lower() != q1.lower():  # Avoid duplicate
        queries.append(q2)
    
    # Q3: Top 2 entities + top 2 seed terms (C)
    if seed_terms:
        q3_parts = top_entities[:2] + seed_terms[:2]
        q3 = " ".join(q3_parts)
        if q3.lower() not in {q.lower() for q in queries}:
            queries.append(q3)
    
    # Q4: Guideline token for high-risk falsifiable claims (C - optional)
    # Check if any claim is high risk with falsifiable_by suggesting protocols
    high_risk_claims = [
        c for c in claims 
        if (c.get("harm_potential", 0) or 0) >= 4
        or c.get("claim_category", "").upper() in ("MEDICAL", "HEALTH", "SAFETY")
    ]
    if high_risk_claims and len(queries) < 4:
        # Add fixed vocabulary token for guideline search
        q4_parts = top_entities[:2] + ["guideline", "recommendation"]
        q4 = " ".join(q4_parts)
        if q4.lower() not in {q.lower() for q in queries}:
            queries.append(q4)
    
    # Ensure minimum 2 queries when entities exist (C)
    if len(queries) < 2 and len(top_entities) >= 2:
        # Variation with fewer entities
        q_fallback = " ".join(top_entities[:2])
        if q_fallback.lower() not in {q.lower() for q in queries}:
            queries.append(q_fallback)
    
    # If still < 2, add entity-only variant
    if len(queries) < 2 and top_entities:
        q_simple = " ".join(top_entities[:4])
        if q_simple.lower() not in {q.lower() for q in queries}:
            queries.append(q_simple)
    
    # Deduplicate, trim to 80 chars, and limit to 4 (C)
    final_queries: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        q_trimmed = q[:80].strip()
        if q_trimmed and q_trimmed.lower() not in seen:
            final_queries.append(q_trimmed)
            seen.add(q_trimmed.lower())
        if len(final_queries) >= 4:
            break
    
    # Calculate anchor count for trace
    anchor_count_used = 0
    if date_anchor:
        anchor_count_used += 1
    if numeric_anchor:
        anchor_count_used += 1

    Trace.event("retrieval.doc_plan", {
        "queries": final_queries,
        "entity_count_used": len(top_entities),
        "anchor_count_used": anchor_count_used,
        "entity_top": top_entities[:8],
    })
    
    return final_queries

# --- Helper: Retrieval Flow ---

async def _execute_retrieval_flow(
    queries: List[str],
    sanity_terms: Set[str],
    search_mgr: Any,
    search_params: Dict[str, Any],
    trace_event_prefix: str = "retrieval.doc"
) -> Tuple[List[EvidenceItem], List[EvidenceSourceMeta], List[str]]:
    """
    Core retrieval logic: Search -> Sanity Gate -> Cluster -> Extract.
    Returns (items, meta, kept_urls).
    """
    RELEVANCE_THRESHOLD = 0.5 
    CLUSTER_TAU = 0.6 
    
    kept_results: List[Dict[str, Any]] = []
    
    # 1. Search (Async)
    tasks = []
    for q in queries:
        tasks.append(search_mgr.search_phase(
            q, 
            max_results=search_params.get("max_results", 5), 
            depth=search_params.get("depth", "basic"), 
            topic=search_params.get("topic", "general")
        ))
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, res in enumerate(responses):
        query = queries[i]
        if isinstance(res, Exception):
            logger.warning(f"Search failed for query '{query}': {res}")
            continue
            
        _, sources = res 
        
        # 2. Sanity Gate
        query_kept_count = 0
        best_score = 0.0
        max_overlap = 0
        
        for src in sources:
            title = src.get("title", "")
            snippet = src.get("content", "") 
            if not snippet:
                snippet = src.get("snippet", "")
                
            text_to_check = f"{title} {snippet}"
            src_tokens = _normalize_text(text_to_check)
            overlap = len(sanity_terms & src_tokens)
            score = float(src.get("score", 0.0) or 0.0)
            
            max_overlap = max(max_overlap, overlap)
            best_score = max(best_score, score)
            
            if overlap >= 1 or score >= RELEVANCE_THRESHOLD:
                src["_temp_text"] = text_to_check
                src["_temp_score"] = score
                kept_results.append(src)
                query_kept_count += 1
                
        Trace.event(f"{trace_event_prefix}.search", {
            "query": query,
            "results_count": len(sources),
            "kept_count": query_kept_count,
            "max_overlap": max_overlap,
            "best_score": best_score
        })
        
    # 3. Redundancy Clustering
    clusters: List[List[Dict[str, Any]]] = []
    kept_results.sort(key=lambda x: x.get("_temp_score", 0.0), reverse=True)
    
    for res in kept_results:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            sim = _calculate_jaccard(res.get("_temp_text", ""), rep.get("_temp_text", ""))
            if sim >= CLUSTER_TAU:
                cluster.append(res)
                placed = True
                break
        if not placed:
            clusters.append([res])
            
    representatives = [c[0] for c in clusters]
    
    Trace.event(f"{trace_event_prefix}.clusters", {
        "cluster_count": len(clusters),
        "max_cluster_size": max([len(c) for c in clusters]) if clusters else 0,
        "tau": CLUSTER_TAU
    })
    
    # 4. Selective Extraction
    extracted_items: List[EvidenceItem] = []
    extraction_failures = 0
    kept_meta: List[EvidenceSourceMeta] = []
    kept_urls: List[str] = []
    
    fetch_tasks = []
    for rep in representatives:
        url = rep.get("url")
        if url:
             fetch_tasks.append(search_mgr.fetch_url_content(url))
        else:
             fetch_tasks.append(None)
             
    fetched_contents = await asyncio.gather(*[t for t in fetch_tasks if t], return_exceptions=True)
    
    fetch_idx = 0
    for i, rep in enumerate(representatives):
        url = rep.get("url", "")
        kept_urls.append(url)
        
        # Meta is always kept for representatives
        meta = EvidenceSourceMeta(
            url=url,
            title=rep.get("title", ""),
            snippet=rep.get("content", "") or rep.get("snippet", ""),
            score=rep.get("_temp_score", 0.0),
            provider_meta=rep
        )
        kept_meta.append(meta)
        
        if fetch_tasks[i] is None:
            continue
            
        content = fetched_contents[fetch_idx]
        fetch_idx += 1
        
        if isinstance(content, Exception) or not content:
            extraction_failures += 1
            continue
            
        item = EvidenceItem(
            url=url,
            extracted_text=str(content),
            citations=[],
            content_hash=_compute_content_hash(str(content)),
            source_meta=meta
        )
        extracted_items.append(item)
        
    Trace.event(f"{trace_event_prefix}.extract", {
        "reps_count": len(representatives),
        "extracted_count": len(extracted_items),
        "failures_count": extraction_failures
    })
    
    # Also include meta for non-representatives? For now just representatives.
    
    return extracted_items, kept_meta, kept_urls

# --- T006: Document Retrieval ---

async def doc_retrieve_to_pool(
    doc_queries: List[str], 
    sanity_terms: Set[str],
    search_mgr: Any,
    config: Any = None
) -> EvidencePool:
    """
    Executes metadata search, sanity gate, clustering, and selective extraction.
    """
    items, meta, _ = await _execute_retrieval_flow(
        doc_queries, 
        sanity_terms, 
        search_mgr, 
        search_params={"max_results": 5, "depth": "basic", "topic": "general"},
        trace_event_prefix="retrieval.doc"
    )
    
    pool = EvidencePool()
    pool.meta = meta
    pool.add_items(items)
    return pool

# --- T009: Deficit Gate ---

def compute_deficit(claim: Dict[str, Any], bundle: EvidenceBundle) -> Deficit:
    """
    Determines if matched evidence is sufficient or if escalation is needed.
    Rules:
    - deficit if matched_count == 0 -> reason "no_pool_matches"
    - deficit if matched sources appear to be from one redundancy cluster only -> "low_independence"
    """
    reasons = []
    
    # 1. No matches
    if not bundle.matched_items:
        reasons.append("no_pool_matches")
        
    # 2. Low independence (check unique domains or clusters)
    # MVP: bundle.cluster_ids is not fully populated yet in match_claim_to_pool 
    # (requires cluster info in pool which we didn't persist).
    # We'll use URL domains for independence check for now.
    domains = set()
    for item in bundle.matched_items:
        # crude domain extraction
        try:
            from urllib.parse import urlparse
            domain = urlparse(item.url).netloc
            domains.add(domain)
        except Exception:
            pass
            
    if bundle.matched_items and len(domains) < 2:
        # If we have matches but only from 1 domain, it's low independence
        # But wait, if we only found 1 strong match, is that a deficit?
        # Maybe. If it's a high risk claim.
        # For MVP, let's say 1 source is OK unless it's a critical claim.
        # The spec says: "deficit if matched sources appear to be from one redundancy cluster only"
        # Since we extract representatives (1 per cluster), each item IS a cluster representative.
        # So if len(matched_items) == 1, it is "one cluster only".
        if len(bundle.matched_items) == 1:
             reasons.append("low_independence")

    # Severity
    risk_score = float(claim.get("risk_score", 0.5) or 0.5)
    is_deficit = bool(reasons)
    severity = risk_score if is_deficit else 0.0
    
    Trace.event("retrieval.claim.deficit", {
        "claim_id": bundle.claim_id,
        "is_deficit": is_deficit,
        "severity": severity,
        "reason_codes": reasons
    })
    
    return Deficit(is_deficit=is_deficit, reason_codes=reasons, severity=severity)

# --- T010: Claim Escalation ---

async def escalate_claim(
    claim: Dict[str, Any], 
    pool: EvidencePool,
    search_mgr: Any
) -> Tuple[EvidencePool, EvidenceBundle]:
    """
    Performs bounded claim-level search to resolve deficit.
    Updates and returns the pool and the new bundle.
    """
    claim_id = str(claim.get("id") or "unknown")
    
    Trace.event("retrieval.claim.escalation.start", {"claim_id": claim_id})
    
    # Pre-calc signals
    sanity_terms = _normalize_text(" ".join(_extract_entities_from_claim(claim)))
    variants = build_query_variants(claim)
    variant_map = {v.query_id: v.text for v in variants}
    
    ladder = get_escalation_ladder()
    resolved = False
    stop_reason = "max_passes"
    
    for i, pass_config in enumerate(ladder):
        if i >= 4:
            break # Max 4 passes (matches ladder length usually)
        
        # Prepare queries
        pass_queries = []
        for qid in pass_config.query_ids:
            q_text = variant_map.get(qid)
            if q_text:
                pass_queries.append(q_text)
                
        if not pass_queries:
            continue
            
        # Topic selection
        topic, _ = select_topic_from_claim(claim)
        if pass_config.topic:
            topic = pass_config.topic
            
        search_params = {
            "max_results": pass_config.max_results,
            "depth": pass_config.search_depth,
            "topic": topic
        }
        
        # Execute Retrieval
        items, meta, _ = await _execute_retrieval_flow(
            pass_queries,
            sanity_terms,
            search_mgr,
            search_params,
            trace_event_prefix="retrieval.claim.escalation"
        )
        
        Trace.event("retrieval.claim.escalation.pass", {
            "pass_id": pass_config.pass_id,
            "queries": pass_queries,
            "kept_count": len(meta),
            "extracted_count": len(items)
        })
        
        # Merge to Pool
        if items:
            pool.add_items(items)
            # Re-match
            bundle = match_claim_to_pool(claim, pool)
            deficit = compute_deficit(claim, bundle)
            
            if not deficit.is_deficit:
                resolved = True
                stop_reason = "deficit_resolved"
                break
                
    # Final bundle check
    bundle = match_claim_to_pool(claim, pool)
    
    Trace.event("retrieval.claim.escalation.stop", {
        "claim_id": claim_id, 
        "resolved": resolved, 
        "reason": stop_reason
    })
    
    return pool, bundle