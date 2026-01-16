import re
from collections import Counter
from typing import List, Dict, Any, Set, Tuple
import hashlib

from pydantic import BaseModel, Field

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.coverage_anchors import Anchor, AnchorKind
from spectrue_core.verification.claims.coverage_anchors import extract_all_anchors

# Module wiring verification (D) - emitted once at import time
Trace.event("retrieval.cegs.module", {
    "module_path": __name__,
    "file": __file__,
})

# --- Helpers ---

def _normalize_text(text: str) -> Set[str]:
    """Normalize text to set of tokens for overlap checking."""
    return set(re.findall(r"\w{3,}", text.lower()))


def _contains_time(text: str) -> bool:
    """Check if text contains time-like patterns (dates, years, etc.)."""
    # Simple regex patterns for time detection
    time_patterns = [
        r'\b\d{4}\b',  # Years like 2024
        r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',  # Dates like 01/15/2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}',  # Month day
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # Day month
    ]
    for pattern in time_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _contains_number(text: str) -> bool:
    """Check if text contains significant numbers (statistics, percentages, etc.)."""
    # Patterns for significant numbers
    number_patterns = [
        r'\b\d+(?:\.\d+)?%',  # Percentages
        r'\$\d+(?:[,.\d]*\d)?',  # Money
        r'\b\d{1,3}(?:,\d{3})+\b',  # Large numbers with commas
        r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|trillion)\b',  # Named numbers
    ]
    for pattern in number_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def _compute_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _extract_entities_from_claim(claim: Dict[str, Any]) -> List[str]:
    """
    Safely extract entities from a claim dict.
    
    Prioritizes structured entity fields, falls back to seed terms.
    """
    entities = []
    
    # 1. context_entities (highest priority)
    ce = claim.get("context_entities")
    if isinstance(ce, list):
        entities.extend([str(e) for e in ce if isinstance(e, str)])
        
    # 2. subject_entities
    se = claim.get("subject_entities")
    if isinstance(se, list):
        entities.extend([str(e) for e in se if isinstance(e, str)])
        
    # 3. subject field
    subj = claim.get("subject")
    if isinstance(subj, str) and subj:
        entities.append(subj)
    
    # 4. retrieval_seed_terms (fallback when entity fields are empty)
    # These are keyword-like terms often containing entities
    if not entities:
        seed_terms = claim.get("retrieval_seed_terms")
        if isinstance(seed_terms, list):
            # Use first 5 seed terms as entity fallback
            for term in seed_terms[:5]:
                if isinstance(term, str) and len(term) >= 2:
                    entities.append(term)
        
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

# --- T007: Pool Matching ---

def match_claim_to_pool(claim: Dict[str, Any], pool: EvidencePool) -> EvidenceBundle:
    """
    Matches pool evidence to a claim using structural overlap and anchor constraints.
    """
    claim_id = str(claim.get("id") or claim.get("claim_id") or "unknown")
    
    # 1. Build relaxed match terms:
    # claim entities + context entities + document context entities
    match_terms: Set[str] = set()
    match_terms |= set(_extract_entities_from_claim(claim))
    match_terms |= set(claim.get("context_entities", []))
    match_terms |= set(claim.get("document_context_entities", []))
    match_terms |= set(claim.get("anchor_terms", []))
    
    # Normalize all terms to tokens (same as content)
    raw_terms = list(match_terms)
    match_terms = set()
    for t in raw_terms:
         if isinstance(t, str):
             match_terms.update(_normalize_text(t))
    
    # Anchors (for optional bonuses, not hard filters)
    claim_text = claim.get("normalized_text") or claim.get("text") or ""
    claim_anchors = extract_all_anchors(claim_text)
    
    has_time_anchor = any(a.kind == AnchorKind.TIME for a in claim_anchors)
    has_number_anchor = any(a.kind == AnchorKind.NUMBER for a in claim_anchors)
    
    scored_items: List[Tuple[float, EvidenceItem]] = []
    
    for item in pool.items:
        # Check Content
        content_text = f"{item.source_meta.title} {item.source_meta.snippet} {item.extracted_text or ''}"
        content_tokens = _normalize_text(content_text)
        
        # Entity Overlap Score
        base_overlap = len(match_terms & content_tokens)
        
        if base_overlap == 0:
            continue  # still need some semantic tie
        
        score = float(base_overlap)
        
        # Optional bonuses (do NOT filter)
        if has_time_anchor and _contains_time(content_text):
            score += 0.5
        if has_number_anchor and _contains_number(content_text):
            score += 0.5
        
        # Small bonus for source score
        score += (item.source_meta.score * 0.1)
        
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
    
    # Priority: Explicit recent years > First available
    date_anchor = None
    if time_anchors:
        best_year = -1
        best_anchor = None
        for a in time_anchors:
            # Look for 4-digit years 1990-2035
            years = re.findall(r"\b(199\d|20[0-3]\d)\b", a.span_text)
            if years:
                max_y = max(int(y) for y in years)
                if max_y > best_year:
                    best_year = max_y
                    best_anchor = a
        
        # If we found a year, usage it. Otherwise fall back to first (e.g. "November 14")
        if best_anchor:
            date_anchor = best_anchor.span_text
        else:
            date_anchor = time_anchors[0].span_text

    # Filter numeric anchors to avoid years (e.g. 2000, 1999) being treated as quantities in queries
    numeric_anchor = None
    for a in number_anchors:
        txt = a.span_text
        # Skip if it looks like a year (4 digits starting with 19 or 20)
        if re.match(r"^(19|20)\d{2}$", txt):
            continue
        numeric_anchor = txt
        break
    
    queries: List[str] = []
    
    # Q1: Top 3 entities + optional date anchor (C)
    q1_parts = top_entities[:3]
    if date_anchor:
        q1_parts = q1_parts + [date_anchor]
    q1 = " ".join(q1_parts)
    queries.append(q1)
    
    # Q2: Top 3 entities + optional numeric anchor (C)
    # Use strided entity slice if possible for diversity (cover indices 3-5)
    if len(top_entities) >= 5:
        q2_entities = top_entities[3:6]
    elif len(top_entities) > 3:
        q2_entities = top_entities[1:4]
    else:
        q2_entities = top_entities[:3]
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
