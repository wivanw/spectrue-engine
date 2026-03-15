# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Serialize ClaimGraph to a compact JSON-safe payload for report storage.

Enum metadata is stored as numeric codes to reduce payload size.
Code tables are documented in docs/DEEP_MODE.md (claim_graph section).
"""

from __future__ import annotations

from typing import Any

from .edges import EdgeRelation
from .types import GraphResult, TypedEdge

# Stable code tables for report payload (must match docs/DEEP_MODE.md and frontend).
RELATION_CODES: list[EdgeRelation] = [
    EdgeRelation.SUPPORTS,
    EdgeRelation.CONTRADICTS,
    EdgeRelation.DEPENDS_ON,
    EdgeRelation.ELABORATES,
    EdgeRelation.UNRELATED,
]
RELATION_TO_CODE: dict[EdgeRelation, int] = {r: i for i, r in enumerate(RELATION_CODES)}

CLAIM_TYPE_ORDER: list[str] = ["core", "numeric", "timeline", "attribution", "sidefact"]
CLAIM_TYPE_TO_CODE: dict[str, int] = {t: i for i, t in enumerate(CLAIM_TYPE_ORDER)}

# Report storage: cap per-edge text so Firestore docs stay bounded (~4–8 KiB per edge pair).
DEFAULT_MAX_RATIONALE_CHARS = 1500
DEFAULT_MAX_EVIDENCE_CHARS = 1500


def _relation_to_code(relation: EdgeRelation) -> int:
    return RELATION_TO_CODE.get(relation, 4)  # 4 = UNRELATED


def _claim_type_to_code(claim_type: str) -> int:
    return CLAIM_TYPE_TO_CODE.get(
        (claim_type or "core").lower() if isinstance(claim_type, str) else "core",
        0,
    )


def serialize_graph_for_report(
    graph_result: GraphResult,
    claim_id_to_text: dict[str, str],
    claim_id_to_rgba: dict[str, list[float]] | None = None,
    claim_id_to_type: dict[str, str] | None = None,
    *,
    max_rationale_chars: int = DEFAULT_MAX_RATIONALE_CHARS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> dict[str, Any]:
    """
    Build a JSON-serializable claim_graph payload for deep_analysis.

    Nodes and edges use numeric codes for enums (relation, claim_type);
    booleans are 0|1. Reduces report size vs string enums.
    """
    if not graph_result or getattr(graph_result, "disabled", False):
        return {"nodes": [], "edges": []}

    claim_id_to_rgba = claim_id_to_rgba or {}
    claim_id_to_type = claim_id_to_type or {}

    # Node IDs: from pre_meta (authoritative for graph membership)
    node_ids = list(graph_result.pre_meta.keys()) if graph_result.pre_meta else []
    if not node_ids and graph_result.all_ranked:
        node_ids = [r.claim_id for r in graph_result.all_ranked]

    nodes: list[dict[str, Any]] = []
    for cid in node_ids:
        pre = graph_result.pre_meta.get(cid)
        post = graph_result.post_meta.get(cid)
        ranked = graph_result.get_ranked_by_id(cid)

        pre_dict: dict[str, Any] = {}
        if pre:
            pre_dict = {
                "position_rank": pre.position_rank,
                "pos_prior": round(float(pre.pos_prior), 4),
                "support_mass": round(float(pre.support_mass), 4),
                "novelty": round(float(pre.novelty), 4),
                "uncertainty_proxy": round(float(pre.uncertainty_proxy), 4),
                "importance_prior": round(float(pre.importance_prior), 4),
                "harm_prior": round(float(pre.harm_prior), 4),
                "node_prior": round(float(pre.node_prior), 4),
            }

        post_dict: dict[str, Any] = {}
        if post:
            post_dict = {
                "pagerank": round(float(post.pagerank), 4),
                "centrality_rank": post.centrality_rank,
                "selected": 1 if post.selected else 0,
                "selection_gain": round(float(post.selection_gain), 4),
                "selection_cost": round(float(post.selection_cost), 4),
            }

        centrality = float(ranked.centrality_score) if ranked else 0.0
        is_key = 1 if (ranked and ranked.is_key_claim) else 0
        in_struct = float(ranked.in_structural_weight) if ranked else 0.0
        in_contra = float(ranked.in_contradict_weight) if ranked else 0.0

        node: dict[str, Any] = {
            "claim_id": cid,
            "text": claim_id_to_text.get(cid, ""),
            "pre_meta": pre_dict,
            "post_meta": post_dict,
            "centrality": round(centrality, 4),
            "is_key_claim": is_key,
            "in_structural_weight": round(in_struct, 4),
            "in_contradict_weight": round(in_contra, 4),
        }
        if cid in claim_id_to_rgba and claim_id_to_rgba[cid] is not None:
            rgba = claim_id_to_rgba[cid]
            if isinstance(rgba, (list, tuple)) and len(rgba) >= 4:
                node["rgba"] = [round(float(x), 4) for x in rgba[:4]]
        if cid in claim_id_to_type:
            node["claim_type"] = _claim_type_to_code(claim_id_to_type[cid])
        nodes.append(node)

    edges: list[dict[str, Any]] = []
    for edge in graph_result.typed_edges or []:
        if not isinstance(edge, TypedEdge):
            continue
        rationale = (edge.rationale_short or "")[:max_rationale_chars]
        evidence = (edge.evidence_spans or "")[:max_evidence_chars]
        edges.append({
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "relation": _relation_to_code(edge.relation),
            "score": round(float(edge.score), 4),
            "rationale_short": rationale if rationale else None,
            "evidence_spans": evidence if evidence else None,
            "cross_topic": 1 if getattr(edge, "cross_topic", False) else 0,
            "same_section": 1 if getattr(edge, "same_section", False) else 0,
        })

    return {"nodes": nodes, "edges": edges}


def fallback_edges_for_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    When ClaimGraphStep is skipped, build a star of synthetic edges from the first
    key claim (or first node) so 3D tree has structure. relation=3 (elaborates).
    """
    if len(nodes) < 2:
        return []
    key_idx = next(
        (i for i, n in enumerate(nodes) if n.get("is_key_claim") == 1),
        0,
    )
    hub_id = nodes[key_idx].get("claim_id")
    if not hub_id:
        return []
    edges: list[dict[str, Any]] = []
    for n in nodes:
        cid = n.get("claim_id")
        if not cid or cid == hub_id:
            continue
        edges.append({
            "src_id": hub_id,
            "dst_id": cid,
            "relation": RELATION_TO_CODE[EdgeRelation.ELABORATES],
            "score": 0.5,
            "rationale_short": None,
            "evidence_spans": None,
            "cross_topic": 0,
            "same_section": 1,
            "synthetic": 1,
        })
    return edges
