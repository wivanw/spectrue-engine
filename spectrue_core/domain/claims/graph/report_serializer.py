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
    claim_id_to_role: dict[str, str] | None = None,
    claim_id_to_extra: dict[str, dict[str, Any]] | None = None,
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
    claim_id_to_role = claim_id_to_role or {}
    claim_id_to_extra = claim_id_to_extra or {}

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
        if cid in claim_id_to_role:
            node["claim_role"] = claim_id_to_role[cid]
        if cid in claim_id_to_extra:
            node["extra"] = claim_id_to_extra[cid]
        nodes.append(node)

    edges: list[dict[str, Any]] = []
    for edge in graph_result.typed_edges or []:
        if not isinstance(edge, TypedEdge):
            continue
        rationale = (edge.rationale_short or "")[:max_rationale_chars]
        evidence = (edge.evidence_spans or "")[:max_evidence_chars]
        out_edge = {
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "relation": _relation_to_code(edge.relation),
            "score": round(float(edge.score), 4),
            "rationale_short": rationale if rationale else None,
            "evidence_spans": evidence if evidence else None,
            "cross_topic": 1 if getattr(edge, "cross_topic", False) else 0,
            "same_section": 1 if getattr(edge, "same_section", False) else 0,
        }
        reason = getattr(edge, "reason", None) or "sim"
        if reason:
            out_edge["reason"] = reason
        sim_score = getattr(edge, "sim_score", None)
        if sim_score is not None:
            out_edge["sim_score"] = round(float(sim_score), 4)
        edges.append(out_edge)

    # Compute derived classifier tags from numerical metadata distributions.
    # Uses μ ± σ statistical thresholds (outlier detection on each signal).
    _classify_nodes(nodes)

    return {"nodes": nodes, "edges": edges}


def _classify_nodes(nodes: list[dict[str, Any]]) -> None:
    """
    Derive categorical tags from numerical metadata using statistical thresholds.

    Each tag is assigned when a claim's signal exceeds mean + 1 standard deviation
    (or falls below mean - 1σ for low signals). This is standard outlier detection
    on the PageRank / pre-graph / post-graph distributions.

    Tags:
      - load_bearing: high pagerank AND high structural support (hub)
      - disputed: both support_mass and contradict_weight are significant
      - critical_gap: high harm × uncertainty (Bayesian expected loss)
      - redundant: novelty significantly below average
      - orphan: isolated node (low pagerank + low structural weight)
      - blind_spot: high uncertainty + no evidence
    """
    if len(nodes) < 2:
        return

    import math

    def _stats(values: list[float]) -> tuple[float, float]:
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mu = sum(values) / n
        var = sum((v - mu) ** 2 for v in values) / n
        return mu, var ** 0.5

    def _get(node: dict, *keys: str) -> float:
        for k in keys:
            v = node.get(k)
            if v is not None:
                return float(v)
            pre = node.get("pre_meta") or {}
            if k in pre:
                return float(pre[k])
            post = node.get("post_meta") or {}
            if k in post:
                return float(post[k])
        return 0.0

    # Gather signal vectors
    pageranks = [_get(n, "centrality", "pagerank") for n in nodes]
    structs = [_get(n, "in_structural_weight") for n in nodes]
    contras = [_get(n, "in_contradict_weight") for n in nodes]
    supports = [_get(n, "support_mass") for n in nodes]
    novelties = [_get(n, "novelty") for n in nodes]
    uncertainties = [_get(n, "uncertainty_proxy") for n in nodes]
    harms = [_get(n, "harm_prior") for n in nodes]
    ev_counts = [float((_get_extra(n) or {}).get("evidence_count", 0)) for n in nodes]

    # Compute per-signal μ ± σ
    pr_mu, pr_s = _stats(pageranks)
    st_mu, st_s = _stats(structs)
    co_mu, co_s = _stats(contras)
    su_mu, su_s = _stats(supports)
    nv_mu, nv_s = _stats(novelties)
    un_mu, un_s = _stats(uncertainties)
    ha_mu, ha_s = _stats(harms)

    # Bayesian expected loss: risk = harm × uncertainty
    risks = [h * u for h, u in zip(harms, uncertainties)]
    ri_mu, ri_s = _stats(risks)

    for i, n in enumerate(nodes):
        tags: list[str] = []

        pr = pageranks[i]
        st = structs[i]
        co = contras[i]
        su = supports[i]
        nv = novelties[i]
        un = uncertainties[i]
        risk = risks[i]
        ev = ev_counts[i]

        # Load-bearing: high centrality AND high structural weight
        if pr > pr_mu + pr_s and st > st_mu + st_s:
            tags.append("load_bearing")

        # Disputed: both support AND contradiction are above average
        if su > su_mu and co > co_mu + co_s:
            tags.append("disputed")

        # Critical gap: Bayesian expected loss (harm × uncertainty) is outlier
        if risk > ri_mu + ri_s:
            tags.append("critical_gap")

        # Redundant: novelty significantly below average
        if nv_s > 0 and nv < nv_mu - nv_s:
            tags.append("redundant")

        # Orphan: both centrality and structural weight below average
        if pr < pr_mu - pr_s and st < st_mu - st_s:
            tags.append("orphan")

        # Blind spot: high uncertainty + zero evidence
        if un > un_mu + un_s and ev == 0:
            tags.append("blind_spot")

        if tags:
            n["tags"] = tags


def _get_extra(node: dict) -> dict | None:
    return node.get("extra")


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
