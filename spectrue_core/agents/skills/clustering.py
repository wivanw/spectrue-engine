from typing import Union

from spectrue_core.schema import ClaimUnit
from spectrue_core.verification.evidence_pack import SearchResult

from .base_skill import BaseSkill, logger
from .clustering_contract import VALID_STANCES, build_evidence_matrix_instructions, build_evidence_matrix_prompt
from .clustering_parsing import (
    build_claims_lite,
    build_ev_mat_cache_key,
    build_sources_lite,
    exception_fallback_all_context,
    postprocess_evidence_matrix,
)

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

        claims_lite = build_claims_lite(claims)
        sources_lite, unreadable_indices = build_sources_lite(search_results)

        num_sources = len(sources_lite)
        instructions = build_evidence_matrix_instructions(num_sources=num_sources)
        prompt = build_evidence_matrix_prompt(claims_lite=claims_lite, sources_lite=sources_lite)
        cache_key = build_ev_mat_cache_key(claims_lite=claims_lite, sources_lite=sources_lite)

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
            clustered_results, _stats = postprocess_evidence_matrix(
                search_results=search_results,
                claims_lite=claims_lite,
                matrix=matrix,
                unreadable_indices=unreadable_indices,
                valid_scoring_stances=VALID_STANCES,
            )
            return clustered_results
        except Exception as e:
            return exception_fallback_all_context(search_results=search_results, error=e)
