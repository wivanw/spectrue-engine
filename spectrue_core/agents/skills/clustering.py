from typing import Union

from spectrue_core.schema import ClaimUnit
from spectrue_core.verification.evidence_pack import SearchResult
from spectrue_core.utils.trace import Trace

from .base_skill import BaseSkill
from .clustering_contract import VALID_STANCES
from .clustering_parsing import (
    build_claims_lite,
    build_ev_mat_cache_key,
    build_sources_lite,
    exception_fallback_all_context,
    postprocess_evidence_matrix,
)
from .scoring_contract import (
    STANCE_PASS_REFUTE_ONLY,
    STANCE_PASS_SINGLE,
    STANCE_PASS_SUPPORT_ONLY,
    build_stance_matrix_instructions,
    build_stance_matrix_prompt,
)
from spectrue_core.verification.evidence import merge_stance_passes

class ClusteringSkill(BaseSkill):
    """
    M68: Clustering Skill with Evidence Matrix Pattern.
    Maps sources to claims (1:1) with strict relevance gating.
    """

    async def cluster_evidence(
        self,
        claims: list[Union[ClaimUnit, dict]], # Support both M70 ClaimUnit and legacy dict
        search_results: list[dict],
        *,
        stance_pass_mode: str = "single",
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

        claims_lite = build_claims_lite(claims)
        sources_lite, unreadable_indices = build_sources_lite(search_results)

        # M103: Quote contract audit trace
        Trace.event("stance_clustering.quote_audit", {
            "total_sources": len(sources_lite),
            "sources_with_quote": sum(1 for s in sources_lite if s.get("has_quote")),
            "per_source": [
                {
                    "index": s["index"],
                    "has_quote": s.get("has_quote", False),
                    "text_len": len(s.get("text", "")),
                    "fields_present": s.get("fields_present", []),
                }
                for s in sources_lite
            ],
        })

        num_sources = len(sources_lite)
        
        # M105: Batching to prevent context overflow
        STANCE_BATCH_SIZE = 10
        
        cluster_timeout = float(getattr(self.runtime.llm, "cluster_timeout_sec", 35.0) or 35.0)
        cluster_timeout = max(35.0, cluster_timeout)
        
        async def run_pass(pass_type: str, trace_suffix: str) -> list[SearchResult]:
            all_matrix_rows = []
            
            # Process in batches
            for i in range(0, num_sources, STANCE_BATCH_SIZE):
                batch_sources = sources_lite[i : i + STANCE_BATCH_SIZE]
                batch_suffix = f"{trace_suffix}_b{i // STANCE_BATCH_SIZE}"
                
                prompt = build_stance_matrix_prompt(claims_lite=claims_lite, sources_lite=batch_sources)
                batch_cache_key = build_ev_mat_cache_key(claims_lite=claims_lite, sources_lite=batch_sources)
                
                instructions = build_stance_matrix_instructions(
                    num_sources=len(batch_sources),
                    pass_type=pass_type,
                )
                
                try:
                    result = await self.llm_client.call_json(
                        model="gpt-5-nano",
                        input=prompt,
                        instructions=instructions,
                        reasoning_effort="low",
                        cache_key=f"{batch_cache_key}_{batch_suffix}",
                        timeout=cluster_timeout,
                        trace_kind="stance_clustering",
                    )
                    batch_matrix = result.get("matrix", [])
                    # Add to total matrix
                    if batch_matrix:
                        all_matrix_rows.extend(batch_matrix)
                except Exception as e:
                    logger.warning("[Clustering] Batch %d failed: %s", i, e)
                    # Proceed with partial results

            clustered_results, _stats = postprocess_evidence_matrix(
                search_results=search_results,
                claims_lite=claims_lite,
                matrix=all_matrix_rows,
                unreadable_indices=unreadable_indices,
                valid_scoring_stances=VALID_STANCES,
            )
            for r in clustered_results:
                r["pass_type"] = pass_type
                if r.get("stance") == "support":
                    r["quote_span"] = r.get("key_snippet") or (r.get("quote_matches") or [None])[0]
                if r.get("stance") == "refute":
                    r["contradiction_span"] = r.get("key_snippet") or (r.get("quote_matches") or [None])[0]
            return clustered_results

        try:
            if (stance_pass_mode or "").lower() == "two_pass":
                # Note: two_pass is legacy/slow, but we support it with batching internally too
                support_results = await run_pass(STANCE_PASS_SUPPORT_ONLY, "support")
                refute_results = await run_pass(STANCE_PASS_REFUTE_ONLY, "refute")
                return merge_stance_passes(
                    support_results=support_results,
                    refute_results=refute_results,
                    original_sources=search_results,
                )

            single_results = await run_pass(STANCE_PASS_SINGLE, "single")
            return single_results
        except Exception as e:
            return exception_fallback_all_context(search_results=search_results, error=e)
