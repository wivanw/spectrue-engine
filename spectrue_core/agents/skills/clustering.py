from spectrue_core.verification.evidence_pack import Claim, SearchResult
from .base_skill import BaseSkill, logger

class ClusteringSkill(BaseSkill):

    async def cluster_evidence(
        self,
        claims: list[Claim],
        search_results: list[dict],
        *,
        lang: str = "en",
    ) -> list[SearchResult]:
        """
        Cluster search results against claims using GPT-5 Nano.
        """
        if not claims or not search_results:
            return []
        
        # Prepare inputs
        claims_lite = [{"id": c["id"], "text": c["text"]} for c in claims]
        results_lite = []
        for i, r in enumerate(search_results):
            text_preview = (r.get("snippet") or "") + " " + (r.get("content") or r.get("extracted_content") or "")
            results_lite.append({
                "index": i,
                "domain": r.get("domain") or r.get("url"), # Hint
                "title": r.get("title", ""),
                "text": text_preview[:600] # Truncate for token efficiency
            })
        
        prompt = f"""Analyze the search results and match them to the most relevant CLAIM.
Claims:
{claims_lite}

Search Results:
{results_lite}

Task:
For each search result, determine:
1. `claim_id`: Which claim ID (c1, c2...) is this result most relevant to? If none, use null.
2. `stance`: Does this source confirm ("support"), deny ("contradict"), or is it neutral/unclear ("neutral") regarding that claim?
3. `relevance`: 0.0-1.0 score.

Output valid JSON with key "mappings" (array of objects):
{{
  "mappings": [
    {{ "result_index": 0, "claim_id": "c1", "stance": "support", "relevance": 0.9 }},
    {{ "result_index": 1, "claim_id": "c2", "stance": "neutral", "relevance": 0.4 }}
    ...
  ]
}}
"""
        # M49/T184: Use LLMClient with Responses API + 30s timeout
        cluster_timeout = float(getattr(self.runtime.llm, "cluster_timeout_sec", 30.0) or 30.0)
        # Ensure timeout is at least 30s as per T184 requirement
        cluster_timeout = max(30.0, cluster_timeout)

        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a stance clustering assistant. Group search results by claim relevance and stance.",
                reasoning_effort="low",
                timeout=cluster_timeout,
                trace_kind="stance_clustering",
            )
            
            mappings = result.get("mappings", [])
            mapping_dict = {m.get("result_index"): m for m in mappings if isinstance(m, dict)}
            
            clustered_results: list[SearchResult] = []
            
            for i, r in enumerate(search_results):
                m = mapping_dict.get(i, {})
                
                # Determine fields
                cid = m.get("claim_id")
                # validation: cid must exist in claims
                if cid and not any(c["id"] == cid for c in claims):
                    cid = "c1" # Fallback to first claim if invalid ID returned
                if not cid:
                    cid = "c1" # Default catch-all

                res = SearchResult(
                    claim_id=cid,
                    url=r.get("url") or r.get("link") or "",
                    domain=r.get("domain"), # Will be enriched later if missing
                    # ... Copy other fields ...
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
                    published_at=r.get("published_date"),
                    source_type=r.get("source_type", "unknown"), # type: ignore
                    stance=m.get("stance", "neutral"), # type: ignore
                    relevance_score=float(m.get("relevance", r.get("relevance_score", 0.0))),
                    # ...
                    key_snippet=None,
                    quote_matches=[],
                    is_trusted=bool(r.get("is_trusted")),
                    is_duplicate=False, # Handled in Evidence builder
                    duplicate_of=None
                )
                clustered_results.append(res)
                
            return clustered_results
            
        except Exception as e:
            logger.warning("[M48] Stance clustering failed: %s. Using fallback.", e)
            return self._fallback_cluster(claims, search_results)

    def _fallback_cluster(self, claims: list[Claim], search_results: list[dict]) -> list[SearchResult]:
        """Graceful degradation: map all results to first claim with neutral stance."""
        clustered_results = []
        cid = claims[0]["id"] if claims else "c1"
        
        for r in search_results:
            res = SearchResult(
                claim_id=cid,
                url=r.get("url") or r.get("link") or "",
                domain=r.get("domain"),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                content_excerpt=(r.get("content") or r.get("extracted_content") or "")[:1500],
                published_at=r.get("published_date"),
                source_type=r.get("source_type", "unknown"), # type: ignore
                stance="neutral", # type: ignore
                relevance_score=float(r.get("relevance_score", 0.5)),
                key_snippet=None,
                quote_matches=[],
                is_trusted=bool(r.get("is_trusted")),
                is_duplicate=False,
                duplicate_of=None
            )
            clustered_results.append(res)
        return clustered_results
