from __future__ import annotations

import logging
from typing import Any

from spectrue_core.utils.trace import Trace
from .types import ClaimNode, CandidateEdge, EdgeRelation, STRUCTURAL_RELATIONS


logger = logging.getLogger(__name__)


async def type_edges(
    edge_typing_skill: Any | None,
    sim_edges: list[tuple[str, str, float]],
    nodes: list[ClaimNode],
    min_edge_score: float = 0.6,
) -> tuple[list[Any], dict[str, int], dict[str, float], dict[str, float]]:
    """
    Apply edge typing skill to similarity edges and filter by minimum score.
    Returns:
        kept_edges: The successfully typed and filtered edges.
        typed_edges_by_relation: Counters for each relation type.
        structural_in: In-degree weights for structural relations.
        contradict_in: In-degree weights for contradiction relations.
    """
    typed_edges: list[Any] = []
    typed_edges_by_relation: dict[str, int] = {}
    structural_in: dict[str, float] = {}
    contradict_in: dict[str, float] = {}

    if edge_typing_skill and sim_edges:
        node_map = {n.claim_id: n for n in nodes}
        candidates: list[CandidateEdge] = []
        for src, dst, sim_score in sim_edges:
            src_node = node_map.get(src)
            dst_node = node_map.get(dst)
            same_section = (
                src_node.section_id == dst_node.section_id if src_node and dst_node else False
            )
            cross_topic = (
                src_node.topic_key != dst_node.topic_key if src_node and dst_node else False
            )
            candidates.append(
                CandidateEdge(
                    src_id=src,
                    dst_id=dst,
                    reason="sim",
                    sim_score=float(sim_score),
                    same_section=same_section,
                    cross_topic=cross_topic,
                )
            )

        try:
            typed_edges = await edge_typing_skill.type_edges_batch(
                candidates, node_map
            )
        except Exception as exc:
            logger.warning("[M72] Edge typing failed: %s", exc)
            Trace.event("edge_typing.error", {"error": str(exc)[:200]})
            typed_edges = []

    kept_edges = []
    for te in typed_edges or []:
        if not hasattr(te, "relation") or te.relation == EdgeRelation.UNRELATED:
            continue
        if float(te.score) < min_edge_score:
            continue
        kept_edges.append(te)
        typed_edges_by_relation[te.relation.value] = (
            typed_edges_by_relation.get(te.relation.value, 0) + 1
        )
        if te.relation in STRUCTURAL_RELATIONS:
            structural_in[te.dst_id] = structural_in.get(te.dst_id, 0.0) + float(te.score)
        if te.relation == EdgeRelation.CONTRADICTS:
            contradict_in[te.dst_id] = contradict_in.get(te.dst_id, 0.0) + float(te.score)

    return kept_edges, typed_edges_by_relation, structural_in, contradict_in
