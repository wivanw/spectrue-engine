from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import logging

from spectrue_core.utils.trace import Trace
from spectrue_core.graph.claim_graph import build_query_clusters

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class ClaimGraphFlowResult:
    key_claim_ids: set[str]
    graph_result: object | None


async def run_claim_graph_flow(
    claim_graph,
    *,
    claims: list[dict],
    runtime_config,
    progress_callback: ProgressCallback | None,
) -> ClaimGraphFlowResult:
    """
    M72/M73/M81: ClaimGraph build + enrichment + tracing.

    Mutates `claims` in-place (importance boosts + graph signal fields), matching
    existing pipeline behavior.
    """
    key_claim_ids: set[str] = set()
    graph_result = None

    if claim_graph and claims:
        if progress_callback:
            await progress_callback("building_claim_graph")

        try:
            graph_result = await claim_graph.build(claims)

            if not graph_result.disabled:
                key_claim_ids = set(graph_result.key_claim_ids)
                for claim in claims:
                    if claim.get("id") in key_claim_ids:
                        claim["importance"] = min(
                            1.0, claim.get("importance", 0.5) + 0.2
                        )

            Trace.event("claim_graph", graph_result.to_trace_dict())

            if graph_result.disabled:
                logger.debug("[M72] ClaimGraph disabled: %s", graph_result.disabled_reason)
            else:
                logger.debug(
                    "[M72] ClaimGraph: %d key claims identified", len(key_claim_ids)
                )

        except Exception as e:
            logger.warning("[M72] ClaimGraph failed: %s. Fallback to original flow.", e)
            Trace.event("claim_graph", {"enabled": True, "error": str(e)[:100]})
            Trace.event(
                "pipeline.claim_graph.exception",
                {
                    "error": str(e)[:200],
                },
            )

    # M73 Layer 2-3: Claim Enrichment with Graph Signals
    enriched_count = 0
    high_tension_count = 0
    if graph_result and not graph_result.disabled:
        cfg = runtime_config.claim_graph

        if cfg.structural_prioritization_enabled:
            for claim in claims:
                claim_id = claim.get("id")
                if not claim_id:
                    continue

                ranked = graph_result.get_ranked_by_id(claim_id)
                if ranked:
                    claim["graph_centrality"] = ranked.centrality_score
                    claim["graph_structural_weight"] = ranked.in_structural_weight
                    claim["graph_tension_score"] = ranked.in_contradict_weight
                    claim["is_key_claim"] = ranked.is_key_claim
                    metadata = claim.get("metadata")
                    if metadata is not None:
                        metadata.is_key_claim = bool(ranked.is_key_claim)
                    enriched_count += 1

                    if ranked.in_structural_weight > cfg.structural_weight_threshold:
                        claim["importance"] = min(
                            1.0, claim.get("importance", 0.5) + cfg.structural_boost
                        )

                    if (
                        cfg.tension_signal_enabled
                        and ranked.in_contradict_weight > cfg.tension_threshold
                    ):
                        claim["importance"] = min(
                            1.0, claim.get("importance", 0.5) + cfg.tension_boost
                        )
                        high_tension_count += 1

            Trace.event(
                "claim_intelligence",
                {
                    "structural_prioritization_enabled": True,
                    "tension_signal_enabled": cfg.tension_signal_enabled,
                    "claims_enriched": enriched_count,
                    "high_tension_claims": high_tension_count,
                    "key_claims_with_scores": [
                        {
                            "id": c.claim_id,
                            "centrality": round(c.centrality_score, 4),
                            "structural": round(c.in_structural_weight, 2),
                            "tension": round(c.in_contradict_weight, 2),
                        }
                        for c in graph_result.key_claims[:5]
                    ],
                },
            )

            if enriched_count > 0:
                logger.debug(
                    "[M73] Enriched %d claims with graph signals (%d high-tension)",
                    enriched_count,
                    high_tension_count,
                )

    # M73 Layer 4: Evidence-Need Routing Tracing
    if runtime_config.claim_graph.evidence_need_routing_enabled and claims:
        evidence_need_dist: dict[str, int] = {}
        for claim in claims:
            need = claim.get("evidence_need", "unknown")
            evidence_need_dist[need] = evidence_need_dist.get(need, 0) + 1

        Trace.event(
            "evidence_need_routing",
            {
                "enabled": True,
                "distribution": evidence_need_dist,
                "sample": [
                    {"id": c.get("id"), "evidence_need": c.get("evidence_need", "unknown")}
                    for c in claims[:3]
                ],
            },
        )

    # Graph-aware query grouping (cluster mapping)
    if claims:
        clusters = build_query_clusters(claims)
        for cluster_id, claim_ids in clusters.items():
            for claim in claims:
                if claim.get("id") in claim_ids:
                    claim["cluster_id"] = cluster_id
        Trace.event(
            "claim_query_clusters",
            {
                "cluster_count": len(clusters),
                "sample_clusters": list(clusters.items())[:3],
            },
        )

    return ClaimGraphFlowResult(key_claim_ids=key_claim_ids, graph_result=graph_result)
