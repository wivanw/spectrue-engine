from __future__ import annotations

import math
from typing import TYPE_CHECKING
from .types import ClaimNode, ClaimPreGraphMeta, RankedClaim

if TYPE_CHECKING:
    from spectrue_core.runtime_config import ClaimGraphConfig


def compute_pre_metadata(
    config: "ClaimGraphConfig",
    nodes: list[ClaimNode],
    knn_map: dict[str, list[tuple[str, float]]],
    position_map: dict[str, int],
) -> dict[str, ClaimPreGraphMeta]:
    """Compute pre-graph priors from kNN similarities."""
    pre_meta: dict[str, ClaimPreGraphMeta] = {}
    gamma = float(config.pos_prior_gamma)
    w_pos = float(config.w_pos)
    w_supp = float(config.w_supp)
    w_imp = float(config.w_imp)
    w_harm = float(config.w_harm)

    importance_lookup = {n.claim_id: n.importance for n in nodes}
    harm_lookup = {n.claim_id: n.harm_potential for n in nodes}

    for node in nodes:
        neighbors = knn_map.get(node.claim_id, [])
        sims = [max(0.0, s) for _, s in neighbors]
        support_mass = sum(sims)
        novelty = 1.0 - max(sims) if sims else 1.0
        if support_mass > 0:
            probs = [s / support_mass for s in sims if s > 0]
            uncertainty = -sum(p * math.log(p) for p in probs if p > 0)
        else:
            uncertainty = 0.0

        pos_rank = position_map.get(node.claim_id, 1)
        pos_prior = math.exp(-gamma * float(pos_rank))
        importance_prior = float(importance_lookup.get(node.claim_id, 0.0))
        harm_prior = float(harm_lookup.get(node.claim_id, 0.0)) / 5.0
        node_prior = max(
            0.0,
            w_pos * pos_prior
            + w_supp * support_mass
            + w_imp * importance_prior
            + w_harm * harm_prior,
        )

        pre_meta[node.claim_id] = ClaimPreGraphMeta(
            claim_id=node.claim_id,
            position_rank=pos_rank,
            pos_prior=pos_prior,
            support_mass=support_mass,
            novelty=novelty,
            uncertainty_proxy=uncertainty,
            importance_prior=importance_prior,
            harm_prior=harm_prior,
            node_prior=node_prior,
        )
    return pre_meta


def build_costs(claims: list[dict], default: float) -> tuple[dict[str, float], dict]:
    """
    Build deterministic cost map with guaranteed coverage.
    Missing/invalid costs fall back to default (>=1).
    """
    cost_map: dict[str, float] = {}
    source = "provided"
    missing_costs = 0
    invalid_costs = 0
    fallback = max(float(default or 0.0), 1.0)
    for c in claims:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        if "cost_estimate" in c:
            try:
                val = float(c.get("cost_estimate") or 0.0)
                if val > 0:
                    cost_map[cid] = val
                    continue
                invalid_costs += 1
            except Exception:
                invalid_costs += 1
        else:
            missing_costs += 1
        cost_map[cid] = fallback
        source = "default_claim_cost"

    return cost_map, {
        "source": source,
        "missing_costs": missing_costs,
        "invalid_costs": invalid_costs,
    }


def build_ranked(
    node_ids: list[str],
    pr_scores: dict[str, float],
    selected: list[str],
    *,
    structural_in: dict[str, float] | None = None,
    contradict_in: dict[str, float] | None = None,
) -> list[RankedClaim]:
    """
    Build final ranked claims array.
    """
    ranked: list[RankedClaim] = []
    structural_in = structural_in or {}
    contradict_in = contradict_in or {}
    for cid in sorted(node_ids, key=lambda x: pr_scores.get(x, 0.0), reverse=True):
        ranked.append(
            RankedClaim(
                claim_id=cid,
                centrality_score=pr_scores.get(cid, 0.0),
                in_structural_weight=structural_in.get(cid, 0.0),
                in_contradict_weight=contradict_in.get(cid, 0.0),
                is_key_claim=cid in selected,
            )
        )
    return ranked
