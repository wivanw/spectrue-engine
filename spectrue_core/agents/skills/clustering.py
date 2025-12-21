from spectrue_core.verification.evidence_pack import SearchResult
from spectrue_core.utils.trace import Trace
from spectrue_core.schema import ClaimUnit
from .base_skill import BaseSkill, logger
import json
import hashlib
from typing import Union

class ClusteringSkill(BaseSkill):
    """
    M68: Clustering Skill with Evidence Matrix Pattern.
    Maps sources to claims (1:1) with strict relevance gating.
    """

    async def cluster_evidence(
        self,
        claims: list[Union[ClaimUnit, dict]], # Support both M70 ClaimUnit and legacy dict
        search_results: list[dict],
    ) -> list[SearchResult]:
        """
        Cluster search results against claims using Evidence Matrix pattern.
        
        Refactoring M68/M70:
        - Maps each source to BEST claim (1:1)
        - M70: Maps to specific `assertion_key` within the claim
        - Gates by relevance < 0.4 (DROP)
        - Filters IRRELEVANT/MENTION stances
        - Extracts quotes directly via LLM
        """
        if not claims or not search_results:
            return []
        
        # 1. Prepare inputs for LLM
        # Handle both ClaimUnit and legacy dicts
        claims_lite = []
        for c in claims:
            if isinstance(c, ClaimUnit):
                # M70 Structured Claim
                assertions_lite = [
                    {"key": a.key, "value": str(a.value)[:50]} 
                    for a in c.assertions
                ]
                claims_lite.append({
                    "id": c.id,
                    "text": c.normalized_text or c.text,
                    "assertions": assertions_lite
                })
            else:
                # Legacy dict
                claims_lite.append({
                    "id": c.get("id"),
                    "text": c.get("text"),
                    "assertions": [] # No assertions for legacy
                })

        results_lite = []
        # M73.5: Track unreadable sources for penalty
        unreadable_indices: set[int] = set()
        UNREADABLE_MARKERS = ["access denied", "403 forbidden", "404 not found", 
                              "javascript required", "please enable javascript",
                              "cloudflare", "captcha", "robot check"]
        
        for i, r in enumerate(search_results):
            # M70: Handle content status hints
            status_hint = ""
            if r.get("content_status") == "unavailable":
                status_hint = "[CONTENT UNAVAILABLE - JUDGE BY SNIPPET/TITLE]"
            
            raw_content = r.get("content") or r.get("extracted_content") or ""
            snippet = r.get("snippet") or ""
            text_preview = f"{snippet} {raw_content}".strip()
            
            # ─────────────────────────────────────────────────────────────────
            # M73.5: NO CONTENT GUARDRAIL
            # If content is too short or contains access denial markers,
            # mark as UNREADABLE and cap relevance at 0.1 in post-processing.
            # ─────────────────────────────────────────────────────────────────
            is_unreadable = False
            
            # Rule 1: Content too short (less than 100 chars)
            if len(raw_content) < 100:
                if not snippet or len(snippet) < 50:
                    is_unreadable = True
                    status_hint = "[UNREADABLE: Content too short]"
            
            # Rule 2: Access denial markers
            content_lower = text_preview.lower()
            for marker in UNREADABLE_MARKERS:
                if marker in content_lower:
                    is_unreadable = True
                    status_hint = f"[UNREADABLE: {marker.upper()}]"
                    break
            
            if is_unreadable:
                unreadable_indices.add(i)
            
            results_lite.append({
                "index": i,
                "domain": r.get("domain") or r.get("url"), 
                "title": r.get("title", ""),
                "text": f"{status_hint} {text_preview[:800]}".strip()
            })
            
        # 2. Build Prompt (Evidence Matrix Pattern)
        instructions = """You are an Evidence Analyst. 
Your task is to map each Search Source to its BEST matching Claim AND Assertion.

## Methodology
1. **1:1 Mapping**: Each Source must be mapped to EXACTLY ONE Claim (the most relevant one).
   - If a source supports/refutes a specific ASSERTION (e.g. location, time, amount), map it to that `assertion_key`.
   - If it covers the whole claim generally, leave `assertion_key` null.
   - CRITICAL: You MUST include an entry in the matrix for EVERY source index.
2. **Stance Classification**:
   - `SUPPORT`: Source confirms the claim/assertion is TRUE.
   - `REFUTE`: Source proves the claim/assertion is FALSE.
   - `MIXED`: Source says it's complicated / partially true.
   - `MENTION`: Topic is mentioned but no clear verdict.
   - `IRRELEVANT`: Source is unrelated to any claim.
3. **Relevance Scoring**: Assign `relevance` (0.0-1.0).
   - If relevance < 0.4, you MUST mark stance as `IRRELEVANT`.
   - If content is [UNAVAILABLE], judge relevance based on title/snippet. Do NOT penalize relevance just because content is missing if the source seems authoritative.
4. **Quote Extraction**: Extract the key text segment that justifies your verdict.

## Output JSON Schema
```json
{
  "matrix": [
    {
      "source_index": 0,
      "claim_id": "c1",
      "assertion_key": "event.location.city", // or null
      "stance": "SUPPORT",
      "relevance": 0.9,
      "quote": "Direct quote from text...",
      "reason": "Explain why..."
    }
  ]
}
```"""

        prompt = f"""Build the Evidence Matrix for these sources.

CLAIMS:
{json.dumps(claims_lite, indent=2)}

SOURCES:
{json.dumps(results_lite, indent=2)}

Return the result in JSON format with key "matrix".
"""
        
        # Calculate cache key based on content hash (M68 Requirement: Stable SHA256)
        content_hash = hashlib.sha256(
            (json.dumps(claims_lite, sort_keys=True) + 
             json.dumps(results_lite, sort_keys=True)).encode()
        ).hexdigest()[:32] 
        
        cache_key = f"ev_mat_v1_{content_hash}"

        # Timeout configuration
        cluster_timeout = float(getattr(self.runtime.llm, "cluster_timeout_sec", 35.0) or 35.0)
        cluster_timeout = max(35.0, cluster_timeout)

        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                cache_key=cache_key,
                timeout=cluster_timeout,
                trace_kind="evidence_matrix",
            )
            
            matrix = result.get("matrix", [])
            
            # 3. Post-Processing Router
            clustered_results: list[SearchResult] = []
            stats = {
                "input_sources": len(search_results),
                "dropped_irrelevant": 0,
                "dropped_mention": 0,
                "dropped_bad_id": 0,
                "dropped_no_quote": 0,
                "dropped_missing": 0,
                "kept_valid": 0
            }
            
            VALID_STANCES = {"SUPPORT", "REFUTE", "MIXED"}
            
            # Map by source_index for easy lookup
            matrix_map = {m.get("source_index"): m for m in matrix if isinstance(m, dict)}
            valid_claim_ids = {c["id"] for c in claims}
            
            for i, r in enumerate(search_results):
                match = matrix_map.get(i)
                
                # Rule 1: Missing Match (LLM broken constraint)
                if not match:
                    stats["dropped_missing"] += 1
                    continue
                
                cid = match.get("claim_id")
                akey = match.get("assertion_key")
                stance = (match.get("stance") or "IRRELEVANT").upper()
                relevance = float(match.get("relevance", 0.0))
                quote = match.get("quote")
                
                # ─────────────────────────────────────────────────────────────
                # M73.5: UNREADABLE PENALTY - Cap relevance at 0.1 for bad sources
                # ─────────────────────────────────────────────────────────────
                if i in unreadable_indices:
                    relevance = min(relevance, 0.1)
                    stats["dropped_unreadable"] = stats.get("dropped_unreadable", 0) + 1
                
                # Rule 2: Invalid Claim ID / Null
                if not cid or cid not in valid_claim_ids:
                    stats["dropped_bad_id"] += 1
                    # DO NOT fallback to c1 (Strict M68)
                    continue
                
                # Rule 3: Relevance Gate
                if relevance < 0.4:
                    stats["dropped_irrelevant"] += 1
                    continue
                
                # Rule 4: Stance Filter (Whitelist)
                if stance not in VALID_STANCES:
                    if stance == "MENTION":
                        stats["dropped_mention"] += 1
                    else:
                        stats["dropped_irrelevant"] += 1
                    continue
                    
                # Rule 5: Quote Mandatory
                if not quote or not str(quote).strip():
                    stats["dropped_no_quote"] += 1
                    continue

                stats["kept_valid"] += 1

                # Construct SearchResult
                res = SearchResult(
                    claim_id=cid, # type: ignore
                    url=r.get("url") or r.get("link") or "",
                    domain=r.get("domain"),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
                    published_at=r.get("published_date"),
                    source_type=r.get("source_type", "unknown"), # type: ignore
                    
                    # M68: Injected Fields
                    stance=stance, # type: ignore
                    relevance_score=relevance,
                    quote_matches=[quote],
                    key_snippet=quote,
                    
                    is_trusted=bool(r.get("is_trusted")),
                    is_duplicate=False,
                    duplicate_of=None,
                    
                    # M70 Fields
                    assertion_key=akey,
                    content_status=r.get("content_status", "available"),
                )
                clustered_results.append(res)
            
            # 4. Telemetry
            Trace.event("evidence.synthesis_stats", stats)
            logger.info("[Clustering] Matrix stats: %s", stats)
                
            return clustered_results
            
        except Exception as e:
            logger.warning("[Clustering] ⚠️ Evidence Matrix LLM failed: %s. Returning empty evidence.", e)
            return []
