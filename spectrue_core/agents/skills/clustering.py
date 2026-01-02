# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import logging
from typing import Union

from spectrue_core.schema import ClaimUnit
from spectrue_core.verification.evidence.evidence_pack import SearchResult
from spectrue_core.utils.trace import Trace

from spectrue_core.verification.evidence.evidence import merge_stance_passes
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

logger = logging.getLogger(__name__)

class ClusteringSkill(BaseSkill):
    """
    Clustering Skill with Evidence Matrix Pattern.
    Maps sources to claims (1:1) with strict relevance gating.
    """

    async def cluster_evidence(
        self,
        claims: list[Union[ClaimUnit, dict]], # Support both ClaimUnit and legacy dict
        search_results: list[dict],
        *,
        stance_pass_mode: str = "single",
    ) -> list[SearchResult]:
        """
        Cluster search results against claims using Evidence Matrix pattern.
        
        Refactoring M68/M70:
        - Maps each source to BEST claim (1:1)
        - Maps to specific `assertion_key` within the claim
        - Gates by relevance < 0.4 (DROP)
        - Filters IRRELEVANT/MENTION stances
        - Extracts quotes directly via LLM
        """
        if not claims or not search_results:
            return []

        claims_lite = build_claims_lite(claims)
        sources_lite, unreadable_indices = build_sources_lite(search_results)

        # Quote contract audit trace
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

        # M117: Log claims passed to clustering for debugging
        Trace.event("stance_clustering.claims_input", {
            "claim_count": len(claims_lite),
            "claim_ids": [c.get("id") for c in claims_lite],
            "claim_texts": [c.get("text", "")[:50] for c in claims_lite],
        })

        num_sources = len(sources_lite)

        # Batching to prevent context overflow
        STANCE_BATCH_SIZE = 10

        cluster_timeout = float(getattr(self.runtime.llm, "cluster_timeout_sec", 35.0) or 35.0)
        cluster_timeout = max(35.0, cluster_timeout)

        async def run_pass(pass_type: str, trace_suffix: str) -> list[SearchResult]:
            all_matrix_rows = []

            # Process in batches
            for i in range(0, num_sources, STANCE_BATCH_SIZE):
                raw_batch = sources_lite[i : i + STANCE_BATCH_SIZE]
                batch_sources = [
                    {**src, "index": local_idx}
                    for local_idx, src in enumerate(raw_batch)
                ]
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
                        for row in batch_matrix:
                            if isinstance(row, dict) and isinstance(row.get("source_index"), int):
                                row["source_index"] = row["source_index"] + i
                        all_matrix_rows.extend(batch_matrix)
                except Exception as e:
                    logger.warning("[Clustering] Batch %d failed: %s", i, e)
                    # Proceed with partial results

            # M117: Log LLM output for debugging claim assignment
            Trace.event("stance_clustering.matrix_output", {
                "total_rows": len(all_matrix_rows),
                "claim_ids_assigned": list(set(
                    r.get("claim_id") for r in all_matrix_rows if isinstance(r, dict)
                )),
                "rows_sample": [
                    {"source_index": r.get("source_index"), "claim_id": r.get("claim_id"), "stance": r.get("stance")}
                    for r in all_matrix_rows[:5] if isinstance(r, dict)
                ],
            })

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

            # FIX: Restore pre-verified stances (e.g. from PhaseRunner shortcuts)
            # If Clustering LLM degraded them to CONTEXT, force restore them.
            if single_results and len(single_results) == len(search_results):
                count_restored = 0
                # Assuming index alignment is preserved
                for i, res in enumerate(single_results):
                     original = search_results[i]
                     pre_stance = (original.get("stance") or "").upper()
                     curr_stance = (res.get("stance") or "").upper()

                     # Only restore if it was explicit SUPPORT/REFUTE and Clustering dropped it
                     if pre_stance in ("SUPPORT", "REFUTE") and curr_stance not in ("SUPPORT", "REFUTE"):
                         final_stance = pre_stance.lower()
                         res["stance"] = final_stance
                         # Boost relevance if needed
                         res["relevance_score"] = max(res.get("relevance_score", 0), 0.85)
                         # If no quote found by LLM, use snippet as backup quote
                         if not res.get("quote") and not res.get("quote_span"):
                              res["quote"] = (original.get("snippet") or "")[:500]
                              if final_stance == "support":
                                  res["quote_span"] = res["quote"]
                              elif final_stance == "refute":
                                  res["contradiction_span"] = res["quote"]
                         count_restored += 1

                     # FIX: Always propagate is_primary flag if present (even if stance wasn't restored)
                     # The Scoring layer needs this to trigger the "[PRIMARY SOURCE]" prompt injection.
                     if original.get("is_primary"):
                         res["is_primary"] = True
                         res["source_type"] = "primary"

                if count_restored > 0:
                     logger.info("[Clustering] Restored %d pre-verified stances (overwrote LLM context)", count_restored)

            return single_results
        except Exception as e:
            return exception_fallback_all_context(search_results=search_results, error=e)
