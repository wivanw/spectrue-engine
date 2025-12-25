from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import logging

from spectrue_core.utils.trace import Trace

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

    # M81/T10: Graph worthiness gate
    graph_worthy = True
    graph_skip_reason = None

    if claim_graph and claims and len(claims) >= 2:
        core_support_count = 0
        unique_topics = set()
        
        # M105: Role alias mapping (claim extraction uses thesis/background, graph expects core/support)
        ROLE_ALIASES = {"thesis": "core", "background": "support"}
        
        for c in claims:
            # Check both claim_role (M80+) and type (legacy) fields
            role = c.get("claim_role") or c.get("type", "core")
            role = ROLE_ALIASES.get(role, role)  # Apply alias
            if role in ("core", "support"):
                core_support_count += 1
            topic = c.get("topic_group", "Other")
            unique_topics.add(topic)

        if core_support_count < 2:
            graph_worthy = False
            graph_skip_reason = "insufficient_core_support_claims"
        elif len(unique_topics) < 2 and len(claims) > 3:
            graph_worthy = False
            graph_skip_reason = "no_topic_diversity"

        if not graph_worthy:
            logger.info(
                "[M81] Skipping ClaimGraph: %s (core_support=%d, topics=%d)",
                graph_skip_reason,
                core_support_count,
                len(unique_topics),
            )
            Trace.event(
                "claim_graph.skipped",
                {
                    "reason": graph_skip_reason,
                    "core_support_count": core_support_count,
                    "unique_topics": len(unique_topics),
                    "total_claims": len(claims),
                },
            )

    if claim_graph and claims and graph_worthy:
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
                logger.info("[M72] ClaimGraph disabled: %s", graph_result.disabled_reason)
            else:
                logger.info(
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
                logger.info(
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

    return ClaimGraphFlowResult(key_claim_ids=key_claim_ids, graph_result=graph_result)
