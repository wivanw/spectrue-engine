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
        num_sources = len(results_lite)
        instructions = f"""You are an Evidence Analyst. 
Your task is to map each Search Source to its BEST matching Claim AND Assertion.

## CRITICAL CONTRACT
- You MUST output EXACTLY {num_sources} matrix rows, one for each source_index from 0 to {num_sources - 1}.
- NEVER return an empty matrix. If unsure about a source, output a row with stance="CONTEXT".
- Every row MUST include: source_index, claim_id (or null), stance, quote (string or null), and optional assertion_key.

## Methodology
1. **1:1 Mapping**: Each Source must be mapped to EXACTLY ONE Claim (the most relevant one).
   - If a source supports/refutes a specific ASSERTION (e.g. location, time, amount), map it to that `assertion_key`.
   - If it covers the whole claim generally, leave `assertion_key` null.
2. **Stance Classification**:
   - `SUPPORT`: Source confirms the claim/assertion is TRUE.
   - `REFUTE`: Source proves the claim/assertion is FALSE.
   - `MIXED`: Source says it's complicated / partially true.
   - `MENTION`: Topic is mentioned but no clear verdict.
   - `CONTEXT`: Source is background/tangentially related but not evidence.
   - `IRRELEVANT`: Source is completely unrelated to any claim.
3. **Relevance Scoring**: Assign `relevance` (0.0-1.0).
   - If relevance < 0.4, you MUST mark stance as `IRRELEVANT` or `CONTEXT`.
   - If content is [UNAVAILABLE], judge relevance based on title/snippet. Do NOT penalize relevance just because content is missing if the source seems authoritative.
4. **Quote Extraction**:
   - For `SUPPORT`, `REFUTE`, `MIXED`: quote MUST be non-empty and directly relevant.
   - For `CONTEXT`, `IRRELEVANT`, `MENTION`: quote can be null or empty string.

## Output JSON Schema
```json
{{
  "matrix": [
    {{
      "source_index": 0,
      "claim_id": "c1",
      "assertion_key": "event.location.city", // or null
      "stance": "SUPPORT",
      "relevance": 0.9,
      "quote": "Direct quote from text...",
      "reason": "Explain why..."
    }}
  ]
}}
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
                trace_kind="stance_clustering",
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
                "dropped_missing": 0,  # NOTE: M78 makes this 0 via fallback
                "kept_scored": 0,      # M78: Sources with scoring stances (SUPPORT/REFUTE/MIXED)
                "kept_context": 0,     # M78: Sources with context stances (CONTEXT)
                "context_fallback": 0,  # M78: Count of sources converted to CONTEXT by fallback

            }
            
            VALID_STANCES = {"SUPPORT", "REFUTE", "MIXED", "CONTEXT"}
            
            # Map by source_index for easy lookup
            matrix_map = {m.get("source_index"): m for m in matrix if isinstance(m, dict)}
            valid_claim_ids = {c["id"] for c in claims_lite}
            
            # ─────────────────────────────────────────────────────────────────
            # M78.1 T3: TELEMETRY — Detect matrix failures
            # ─────────────────────────────────────────────────────────────────
            mapped_count = len(matrix_map)
            input_count = len(search_results)
            
            if mapped_count == 0 and input_count > 0:
                # EMPTY matrix — complete LLM failure
                Trace.event("matrix.failure", {
                    "mode": "empty",
                    "input_sources": input_count,
                    "mapped_rows": 0,
                    "claim_count": len(claims_lite),
                })
                logger.warning("[Clustering] ⚠️ Matrix EMPTY: LLM returned 0 rows for %d sources", input_count)
            elif mapped_count < input_count:
                # PARTIAL matrix — some sources unmapped
                Trace.event("matrix.failure", {
                    "mode": "partial",
                    "input_sources": input_count,
                    "mapped_rows": mapped_count,
                    "missing_sources": input_count - mapped_count,
                })
                logger.warning("[Clustering] ⚠️ Matrix PARTIAL: %d/%d sources mapped", mapped_count, input_count)
            
            for i, r in enumerate(search_results):
                match = matrix_map.get(i)
                
                # ─────────────────────────────────────────────────────────────
                # M78.1 T2: SOFT FALLBACK — Unmapped sources become CONTEXT
                # ─────────────────────────────────────────────────────────────
                if not match:
                    # Create synthetic CONTEXT entry (no drop!)
                    stats["context_fallback"] += 1
                    res = SearchResult(
                        claim_id=None,  # type: ignore
                        url=r.get("url") or r.get("link") or "",
                        domain=r.get("domain"),
                        title=r.get("title", ""),
                        snippet=r.get("snippet", ""),
                        content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
                        published_at=r.get("published_date"),
                        source_type=r.get("source_type", "unknown"),  # type: ignore
                        stance="context",  # type: ignore  # Lowercase for schema
                        relevance_score=0.0,
                        quote_matches=[],
                        key_snippet=r.get("snippet", ""),
                        is_trusted=bool(r.get("is_trusted")),
                        is_duplicate=False,
                        duplicate_of=None,
                        assertion_key=None,
                        content_status=r.get("content_status", "available"),
                    )
                    clustered_results.append(res)
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
                
                # Rule 2: Invalid Claim ID / Null (M77 Soft Fallback)
                if not cid or cid not in valid_claim_ids:
                    # M77: Do NOT drop. Keep as CONTEXT.
                    cid = None
                    stance = "CONTEXT"
                    relevance = 0.0
                    akey = None
                    if not quote:
                         quote = r.get("snippet", "")
                
                # Rule 3: Relevance Gate
                if stance not in {"CONTEXT", "IRRELEVANT"} and relevance < 0.4:
                    stats["dropped_irrelevant"] += 1
                    continue
                
                # Rule 4: Stance Filter (Whitelist)
                if stance not in VALID_STANCES:
                    if stance == "MENTION":
                        stats["dropped_mention"] += 1
                    else:
                        stats["dropped_irrelevant"] += 1
                    continue
                    
                # Rule 5: Quote Mandatory (only for scoring stances)
                if stance not in {"CONTEXT", "IRRELEVANT"} and (not quote or not str(quote).strip()):
                    stats["dropped_no_quote"] += 1
                    continue
                # M78: Track scored vs context separately
                if stance in {"SUPPORT", "REFUTE", "MIXED"}:
                    stats["kept_scored"] += 1
                else:  # CONTEXT
                    stats["kept_context"] += 1

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
                    stance=stance.lower(), # type: ignore  # Normalize to lowercase
                    relevance_score=relevance,
                    quote_matches=[quote] if quote else [],
                    key_snippet=quote,
                    
                    is_trusted=bool(r.get("is_trusted")),
                    is_duplicate=False,
                    duplicate_of=None,
                    
                    # M70 Fields
                    assertion_key=akey,
                    content_status=r.get("content_status", "available"),
                )
                clustered_results.append(res)
            
            # ─────────────────────────────────────────────────────────────────
            # M78.1 T3: TELEMETRY — Degraded mode (too many CONTEXT)
            # ─────────────────────────────────────────────────────────────────
            context_count = sum(1 for r in clustered_results if r.get("stance") == "context")
            if clustered_results and context_count / len(clustered_results) > 0.7:
                Trace.event("matrix.degraded", {
                    "context_ratio": round(context_count / len(clustered_results), 2),
                    "threshold": 0.7,
                    "input_sources": input_count,
                })
                logger.warning("[Clustering] ⚠️ Matrix DEGRADED: %.0f%% sources are CONTEXT", 
                              (context_count / len(clustered_results)) * 100)
            
            # 4. Telemetry
            Trace.event("evidence.synthesis_stats", stats)
            logger.info("[Clustering] Matrix stats: %s", stats)
                
            return clustered_results
            
        except Exception as e:
            # ─────────────────────────────────────────────────────────────────
            # M78.1: EXCEPTION FALLBACK — Convert ALL sources to CONTEXT
            # ─────────────────────────────────────────────────────────────────
            logger.warning("[Clustering] ⚠️ Evidence Matrix LLM failed: %s. Converting all to CONTEXT.", e)
            Trace.event("matrix.failure", {
                "mode": "exception",
                "input_sources": len(search_results),
                "error": str(e)[:200],
            })
            
            fallback_results: list[SearchResult] = []
            for r in search_results:
                res = SearchResult(
                    claim_id=None,  # type: ignore
                    url=r.get("url") or r.get("link") or "",
                    domain=r.get("domain"),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
                    published_at=r.get("published_date"),
                    source_type=r.get("source_type", "unknown"),  # type: ignore
                    stance="context",  # type: ignore
                    relevance_score=0.0,
                    quote_matches=[],
                    key_snippet=r.get("snippet", ""),
                    is_trusted=bool(r.get("is_trusted")),
                    is_duplicate=False,
                    duplicate_of=None,
                    assertion_key=None,
                    content_status=r.get("content_status", "available"),
                )
                fallback_results.append(res)
            return fallback_results
