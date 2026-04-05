from __future__ import annotations

from spectrue_core.utils.trace import Trace
from .types import ClaimPreGraphMeta


def trace_pre_metadata(pre_meta: dict[str, ClaimPreGraphMeta]) -> None:
    payload = [
        {
            "id": m.claim_id,
            "pos_prior": round(m.pos_prior, 4),
            "support": round(m.support_mass, 4),
            "novelty": round(m.novelty, 4),
            "uncertainty": round(m.uncertainty_proxy, 4),
            "node_prior": round(m.node_prior, 4),
        }
        for m in pre_meta.values()
    ]
    Trace.event("claim_graph.pre_metadata", {"items": payload})


def trace_knn(knn_map: dict[str, list[tuple[str, float]]], top_k: int) -> None:
    payload = {
        cid: [{"id": nid, "sim": round(sim, 4)} for nid, sim in neigh[:top_k]]
        for cid, neigh in knn_map.items()
    }
    Trace.event("claim_graph.knn", {"neighbors": payload})


def trace_edges(edges: list[tuple[str, str, float]], top_k: int) -> None:
    if not edges:
        Trace.event("claim_graph.edges.sim", {"count": 0})
        return
    weights = [w for _, _, w in edges]
    mean_w = sum(weights) / len(weights)
    top = sorted(edges, key=lambda e: e[2], reverse=True)[:top_k]
    Trace.event(
        "claim_graph.edges.sim",
        {
            "count": len(edges),
            "min": round(min(weights), 4),
            "max": round(max(weights), 4),
            "mean": round(mean_w, 4),
            "top": [(u, v, round(w, 4)) for u, v, w in top],
        },
    )


def trace_mst(mst_edges: list[tuple[str, str, float]], node_count: int, top_k: int) -> None:
    Trace.event(
        "claim_graph.mst",
        {
            "edges": [(u, v, round(w, 4)) for u, v, w in mst_edges[:top_k]],
            "node_count": node_count,
            "connected": len(mst_edges) >= max(0, node_count - 1),
        },
    )


def trace_pagerank(
    pr_scores: dict[str, float],
    pre_meta: dict[str, ClaimPreGraphMeta],
    centrality_rank: dict[str, int],
    degree_weights: dict[str, float],
    top_k: int,
) -> None:
    top_ids = sorted(pr_scores, key=pr_scores.get, reverse=True)[:top_k]
    Trace.event(
        "claim_graph.pagerank",
        {
            "top": [
                {
                    "id": cid,
                    "pagerank": round(pr_scores.get(cid, 0.0), 6),
                    "prior": round(pre_meta.get(cid).node_prior, 6) if cid in pre_meta else 0.0,
                    "rank": centrality_rank.get(cid, -1),
                    "prior_components": {
                        "pos": round(pre_meta[cid].pos_prior, 6) if cid in pre_meta else 0.0,
                        "supp": round(pre_meta[cid].support_mass, 6) if cid in pre_meta else 0.0,
                        "imp": round(pre_meta[cid].importance_prior, 6) if cid in pre_meta else 0.0,
                        "harm": round(pre_meta[cid].harm_prior, 6) if cid in pre_meta else 0.0,
                    },
                    "degree_weight": round(degree_weights.get(cid, 0.0), 6),
                }
                for cid in top_ids
            ]
        },
    )


def trace_selection(
    selected: list[str],
    steps: list[dict],
    budget: float,
    cost_info: dict,
    mode: str,
    top_k: int,
) -> None:
    Trace.event(
        "claim_graph.selection",
        {
            "selected": selected,
            "budget": budget,
            "mode": mode,
            "cost_info": cost_info,
            "steps": [
                {
                    "id": s["id"],
                    "gain": round(float(s["gain"]), 6),
                    "cost": round(float(s["cost"]), 6),
                    "remaining_budget": round(float(s["remaining_budget"]), 6),
                }
                for s in steps[:top_k]
            ],
        },
    )


def trace_quality(scalar: float, info: dict) -> None:
    Trace.event(
        "claim_graph.quality",
        {
            "scalar": round(float(scalar), 4),
            **{k: v for k, v in info.items()},
        },
    )
