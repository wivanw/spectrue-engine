from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Awaitable, Callable

import asyncio
import logging
import math

from spectrue_core.schema.signals import TimeWindow
from spectrue_core.schema.scoring import BeliefState
from spectrue_core.graph.context import ClaimContextGraph
from spectrue_core.graph.propagation import propagate_belief, propagation_routing_signals
from spectrue_core.scoring.belief import (
    calculate_evidence_impact, 
    update_belief, 
    log_odds_to_prob,
    apply_consensus_bound
)
from spectrue_core.scoring.consensus import calculate_consensus
from spectrue_core.verification.temporal import (
    label_evidence_timeliness,
    normalize_time_window,
)
from spectrue_core.verification.source_utils import canonicalize_sources
from spectrue_core.verification.trusted_sources import is_social_platform

# M108: Suppress deprecation warning - full migration to Bayesian scoring is future work
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from spectrue_core.verification.scoring_aggregation import aggregate_claim_verdict

from spectrue_core.verification.rgba_aggregation import (
    apply_dependency_penalties,
    apply_conflict_explainability_penalty,
    recompute_verified_score,
)
from spectrue_core.verification.search_policy import (
    resolve_profile_name,
    resolve_stance_pass_mode,
)
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]


_TIER_PRIOR = {
    "D": 0.80,
    "C": 0.85,
    "B": 0.90,
    "A'": 0.93,
    "A": 0.96,
    "UNKNOWN": 0.90,
}
_TIER_PRIOR_BASELINE = _TIER_PRIOR["B"]


def _explainability_factor_for_tier(tier: str | None) -> tuple[float, str, float]:
    if not tier:
        return _TIER_PRIOR["UNKNOWN"] / _TIER_PRIOR_BASELINE, "unknown_default", _TIER_PRIOR["UNKNOWN"]
    prior = _TIER_PRIOR.get(str(tier).strip().upper(), _TIER_PRIOR["UNKNOWN"])
    return prior / _TIER_PRIOR_BASELINE, "best_tier", prior


def _apply_explainability_tier_cap(
    score: float,
    tier_counts: dict[str, int],
) -> tuple[float, dict]:
    """Apply tier-based explainability cap to a score.

    Args:
        score: The raw explainability score (0..1).
        tier_counts: Dict mapping tier names (A, B, C, D) to counts.

    Returns:
        Tuple of (adjusted_score, trace_dict) where trace_dict contains
        debugging info including best_tier, factor, pre/post values.
    """
    # Find best tier from counts (A > A' > B > C > D)
    tier_priority = ["A", "A'", "B", "C", "D"]
    best_tier: str | None = None
    for tier in tier_priority:
        if tier_counts.get(tier, 0) > 0:
            best_tier = tier
            break

    factor, source, prior = _explainability_factor_for_tier(best_tier)
    post_score = max(0.0, min(1.0, score * factor))

    trace = {
        "best_tier": best_tier,
        "pre_A": score,
        "prior": prior,
        "baseline": _TIER_PRIOR_BASELINE,
        "factor": factor,
        "post_A": post_score,
        "source": source,
    }
    return post_score, trace



@dataclass(frozen=True, slots=True)
class EvidenceFlowInput:
    fact: str
    original_fact: str
    lang: str
    content_lang: str | None
    gpt_model: str
    search_type: str
    progress_callback: ProgressCallback | None
    prior_belief: BeliefState | None = None
    context_graph: ClaimContextGraph | None = None


async def run_evidence_flow(
    *,
    agent,
    search_mgr,
    build_evidence_pack,
    enrich_sources_with_trust,
    inp: EvidenceFlowInput,
    claims: list[dict],
    sources: list[dict],
) -> dict:
    """
    Analysis + scoring: clustering, social verification, evidence pack, scoring, finalize.

    Matches existing pipeline behavior; expects sources/claims to be mutable dict shapes.
    """
    if inp.progress_callback:
        await inp.progress_callback("ai_analysis")

    current_cost = search_mgr.calculate_cost(inp.gpt_model, inp.search_type)

    sources = canonicalize_sources(sources)

    time_windows: dict[str, TimeWindow] = {}
    if claims:
        default_relative_days = getattr(
            getattr(getattr(search_mgr, "config", None), "runtime", None),
            "temporal",
            None,
        )
        default_days = getattr(default_relative_days, "relative_window_days", None)
        default_days = int(default_days) if isinstance(default_days, int) else None

        for claim in claims:
            claim_id = str(claim.get("id") or "c1")
            metadata = claim.get("metadata")
            time_signals = []
            time_sensitive = False
            if metadata:
                time_signals = list(getattr(metadata, "time_signals", []) or [])
                time_sensitive = bool(getattr(metadata, "time_sensitive", False))

            req = claim.get("evidence_requirement") or {}
            if isinstance(req, dict) and req.get("is_time_sensitive"):
                time_sensitive = True
            if isinstance(req, dict) and req.get("needs_recent_source"):
                time_sensitive = True

            if time_signals or time_sensitive:
                time_windows[claim_id] = normalize_time_window(
                    time_signals,
                    reference_date=date.today(),
                    default_relative_days=default_days or 30,
                )
            else:
                time_windows[claim_id] = normalize_time_window(
                    [],
                    reference_date=date.today(),
                    default_relative_days=default_days or 30,
                )

        for claim_id, window in time_windows.items():
            claim_sources = [s for s in sources if s.get("claim_id") == claim_id]
            if not claim_sources and len(time_windows) == 1:
                label_evidence_timeliness(sources, time_window=window)
            else:
                label_evidence_timeliness(claim_sources, time_window=window)

    clustered_results = None
    if claims and sources:
        if inp.progress_callback:
            await inp.progress_callback("clustering_evidence")
        profile_name = resolve_profile_name(inp.search_type)
        stance_pass_mode = resolve_stance_pass_mode(profile_name)
        clustered_results = await agent.cluster_evidence(
            claims,
            sources,
            stance_pass_mode=stance_pass_mode,
        )

    anchor_claim = None
    anchor_claim_id = None
    if claims:
        core_claims = [c for c in claims if c.get("type") == "core"]
        anchor_claim = (
            max(core_claims, key=lambda c: c.get("importance", 0))
            if core_claims
            else claims[0]
        )
        anchor_claim_id = anchor_claim.get("id") or anchor_claim.get("claim_id")

    # M67: Inline Social Verification (Tier A')
    if claims and sources:
        verify_tasks = []
        social_indices = []

        for i, src in enumerate(sources):
            stype = src.get("source_type", "general")
            domain = src.get("domain", "")
            if stype == "social" or is_social_platform(domain):
                anchor = anchor_claim if anchor_claim else claims[0]
                verify_tasks.append(
                    agent.verify_social_statement(
                        anchor,
                        src.get("content", "") or src.get("snippet", ""),
                        src.get("url", "") or src.get("link", ""),
                    )
                )
                social_indices.append(i)

        if verify_tasks:
            if inp.progress_callback:
                await inp.progress_callback("verifying_social")

            social_results = await asyncio.gather(*verify_tasks, return_exceptions=True)

            for idx, res in zip(social_indices, social_results):
                if isinstance(res, dict) and res.get("tier") == "A'":
                    sources[idx]["evidence_tier"] = "A'"
                    sources[idx]["source_type"] = "official"
                    logger.debug(
                        "[M67] Promoted Social Source %s to Tier A'",
                        sources[idx].get("domain"),
                    )

    # T7: Deterministic Ranking
    claims.sort(key=lambda c: (-c.get("importance", 0.0), c.get("text", "")))

    pack = build_evidence_pack(
        fact=inp.original_fact,
        claims=claims,
        sources=sources,
        search_results_clustered=clustered_results,
        content_lang=inp.content_lang or inp.lang,
        article_context={"text_excerpt": inp.fact[:500]}
        if inp.fact != inp.original_fact
        else None,
    )

    if inp.progress_callback:
        await inp.progress_callback("score_evidence")

    # M108: Single batch LLM call for all claims (not per-claim to avoid 6x cost)
    result = await agent.score_evidence(pack, model=inp.gpt_model, lang=inp.lang)

    # M93: Apply dependency penalties after scoring
    claim_verdicts = result.get("claim_verdicts")
    if isinstance(claim_verdicts, list):
        changed = apply_dependency_penalties(claim_verdicts, claims)
        if changed:
            recalculated = recompute_verified_score(claim_verdicts)
            if recalculated is not None:
                result["verified_score"] = recalculated

    conflict_detected = False
    verdict_state_by_claim: dict[str, str] = {}
    importance_by_claim: dict[str, float] = {}
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        cid = str(claim.get("id") or "").strip()
        if not cid:
            continue
        try:
            importance_by_claim[cid] = float(claim.get("importance", 1.0) or 1.0)
        except Exception:
            importance_by_claim[cid] = 1.0
    if isinstance(claim_verdicts, list):
        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            claim_id = str(cv.get("claim_id") or "").strip()
            if not claim_id and claims:
                claim_id = str(claims[0].get("id") or "c1")
                cv["claim_id"] = claim_id

            claim_obj = next((c for c in (claims or []) if c.get("id") == claim_id), None)
            temporality = claim_obj.get("temporality") if isinstance(claim_obj, dict) else None

            items = pack.get("items", []) if isinstance(pack, dict) else []
            has_direct_evidence = False
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if claim_id and item.get("claim_id") not in (None, claim_id):
                        continue
                    stance = str(item.get("stance") or "").upper()
                    if stance in ("SUPPORT", "REFUTE") and item.get("quote"):
                        has_direct_evidence = True
                        break

            agg = None
            if has_direct_evidence:
                Trace.event(
                    "verdict.override",
                    {
                        "claim_id": claim_id,
                        "mode": "deterministic",
                        "reason": "direct_quote_evidence",
                    },
                )
                agg = aggregate_claim_verdict(
                    pack,
                    policy={},
                    claim_id=claim_id,
                    temporality=temporality if isinstance(temporality, dict) else None,
                )
                cv["verdict_score"] = agg.get("verdict_score", 0.5)
                cv["verdict"] = agg.get("verdict", "ambiguous")
                cv["status"] = agg.get("verdict", "ambiguous")
                cv["reasons_expert"] = agg.get("reasons_expert", {})
                cv["reasons_short"] = cv.get("reasons_short", []) or []
            else:
                Trace.event(
                    "verdict.override",
                    {
                        "claim_id": claim_id,
                        "mode": "llm_only",
                        "reason": "no_direct_quote_evidence",
                    },
                )
                agg = aggregate_claim_verdict(
                    pack,
                    policy={},
                    claim_id=claim_id,
                    temporality=temporality if isinstance(temporality, dict) else None,
                )
                cv["reasons_short"] = cv.get("reasons_short", []) or []

            explainability = result.get("explainability_score", -1.0)
            if isinstance(explainability, (int, float)) and explainability >= 0:
                best_tier = agg.get("best_tier") if isinstance(agg, dict) else None
                pre_a = float(explainability)
                factor, source, prior = _explainability_factor_for_tier(best_tier)
                if math.isfinite(pre_a):
                    post_a = max(0.0, min(1.0, pre_a * factor))
                    if abs(post_a - pre_a) > 1e-9:
                        result["explainability_score"] = post_a
                    Trace.event(
                        "verdict.explainability_tier_factor",
                        {
                            "claim_id": claim_id,
                            "best_tier": best_tier,
                            "pre_A": pre_a,
                            "prior": prior,
                            "baseline": _TIER_PRIOR_BASELINE,
                            "factor": factor,
                            "post_A": post_a,
                            "source": source,
                        },
                    )
                else:
                    Trace.event(
                        "verdict.explainability_missing",
                        {"claim_id": claim_id, "best_tier": best_tier},
                    )

            verdict_state = "insufficient_evidence"
            if cv["verdict"] == "verified":
                verdict_state = "supported"
            elif cv["verdict"] == "refuted":
                verdict_state = "refuted"
            elif cv["verdict"] == "ambiguous":
                verdict_state = "conflicted"

            cv["verdict_state"] = verdict_state
            verdict_state_by_claim[claim_id] = verdict_state

            conflict_detected = conflict_detected or bool(
                (cv.get("reasons_expert") or {}).get("conflict")
            )

        changed = apply_dependency_penalties(claim_verdicts, claims)
        recalculated = recompute_verified_score(claim_verdicts)
        if recalculated is not None:
            result["verified_score"] = recalculated
        if changed:
            result["dependency_penalty_applied"] = True

    # M104: Bayesian Scoring
    if inp.prior_belief:
        current_belief = inp.prior_belief
        
        # Consensus Calculation
        evidence_list = []
        raw_evidence = getattr(pack, "evidence", []) if not isinstance(pack, dict) else pack.get("evidence", [])
        
        # Helper to wrap dicts if needed
        class MockEvidence:
            def __init__(self, d):
                self.domain = d.get("domain")
                self.stance = d.get("stance")
                
        for e in raw_evidence:
            if isinstance(e, dict):
                evidence_list.append(MockEvidence(e))
            else:
                evidence_list.append(e)
                
        consensus = calculate_consensus(evidence_list)
        
        # Claim Graph Propagation
        if inp.context_graph and isinstance(claim_verdicts, list):
            for cv in claim_verdicts:
                cid = cv.get("claim_id")
                node = inp.context_graph.get_node(cid)
                if node:
                    v = cv.get("verdict", "ambiguous")
                    conf = cv.get("confidence", 1.0)
                    if not isinstance(conf, (int, float)):
                        conf = 1.0
                    impact = calculate_evidence_impact(v, confidence=conf)
                    node.local_belief = BeliefState(log_odds=impact)
            
            propagate_belief(inp.context_graph)
            result["graph_propagation"] = propagation_routing_signals(inp.context_graph)
            
            # Update from Anchor
            if anchor_claim_id:
                anchor_node = inp.context_graph.get_node(anchor_claim_id)
                if anchor_node and anchor_node.propagated_belief:
                    current_belief = update_belief(current_belief, anchor_node.propagated_belief.log_odds)
        
        elif isinstance(claim_verdicts, list):
            # Fallback: Sum updates (weighted by verdict strength + claim importance)
            for cv in claim_verdicts:
                if not isinstance(cv, dict):
                    continue
                v = cv.get("verdict", "ambiguous")
                cid = str(cv.get("claim_id") or "").strip()
                try:
                    strength = float(cv.get("verdict_score", 0.5) or 0.5)
                except Exception:
                    strength = 0.5
                strength = max(0.0, min(1.0, strength))
                relevance = max(0.0, min(1.0, importance_by_claim.get(cid, 1.0)))
                impact = calculate_evidence_impact(v, confidence=strength, relevance=relevance)
                current_belief = update_belief(current_belief, impact)
                 
        # Apply Consensus
        current_belief = apply_consensus_bound(current_belief, consensus)
        
        # Set Result
        result["verified_score"] = log_odds_to_prob(current_belief.log_odds)
        verified_cap = recompute_verified_score(claim_verdicts) if isinstance(claim_verdicts, list) else None
        if verified_cap is not None and result["verified_score"] > verified_cap:
            result["verified_score"] = verified_cap
        
        # Trace
        result["bayesian_trace"] = {
            "prior_log_odds": inp.prior_belief.log_odds,
            "consensus_score": consensus.score,
            "posterior_log_odds": current_belief.log_odds,
            "final_probability": result["verified_score"]
        }

    if conflict_detected:
        explainability = result.get("explainability_score", -1.0)
        if isinstance(explainability, (int, float)) and explainability >= 0:
            result["explainability_score"] = apply_conflict_explainability_penalty(
                float(explainability),
            )
            audit = result.get("audit") or {}
            explainability_audit = audit.get("explainability", {})
            if isinstance(explainability_audit, dict):
                explainability_audit["conflict_penalty_applied"] = True
                audit["explainability"] = explainability_audit
                result["audit"] = audit


    if inp.progress_callback:
        await inp.progress_callback("finalizing")

    result["cost"] = current_cost
    result["text"] = inp.fact
    result["search_meta"] = search_mgr.get_search_meta()
    display_sources = pack.get("scored_sources")
    if not isinstance(display_sources, list) or not display_sources:
        display_sources = pack.get("context_sources")
    if not isinstance(display_sources, list):
        display_sources = []
    result["sources"] = enrich_sources_with_trust(display_sources)

    if anchor_claim:
        result["anchor_claim"] = {
            "text": anchor_claim.get("text", ""),
            "type": anchor_claim.get("type", "core"),
            "importance": anchor_claim.get("importance", 1.0),
        }
        if anchor_claim_id and anchor_claim_id in verdict_state_by_claim:
            result["verdict_state"] = verdict_state_by_claim[anchor_claim_id]
        if anchor_claim_id and anchor_claim_id in time_windows:
            result["time_window"] = time_windows[anchor_claim_id].to_dict()

    timeliness_labels: list[dict] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        status = src.get("timeliness_status")
        url = src.get("url") or src.get("link")
        if status and url:
            timeliness_labels.append(
                {"source_url": url, "timeliness_status": status}
            )
    if timeliness_labels:
        result["timeliness_labels"] = timeliness_labels
    if result.get("time_window") or timeliness_labels:
        audit = result.get("audit") or {}
        if result.get("time_window"):
            audit["time_window"] = result.get("time_window")
        if timeliness_labels:
            audit["timeliness_labels"] = timeliness_labels
        result["audit"] = audit

    verified = result.get("verified_score", -1.0)
    if verified < 0:
        logger.warning("[Pipeline] ⚠️ Missing verified_score in result - using 0.5")
        verified = 0.5
        result["verified_score"] = verified

    return result
