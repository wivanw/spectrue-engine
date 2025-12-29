from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Awaitable, Callable, Any

import asyncio
import logging
import math

from spectrue_core.utils.embedding_service import EmbedService
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
)
from spectrue_core.verification.claim_selection import pick_ui_main_claim
from spectrue_core.verification.search_policy import (
    resolve_profile_name,
    resolve_stance_pass_mode,
)
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

def _norm_id(x: Any) -> str:
    return str(x or "").strip().lower()

def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and 0.0 <= float(x) <= 1.0

def _logit(p: float) -> float:
    # Contract: p must be in (0,1). No clamping here.
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _claim_text(cv: dict) -> str:
    text = cv.get("claim_text") or cv.get("claim") or cv.get("text") or ""
    return str(text).strip()

def _mark_anchor_duplicates_sync(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict],
    embed_service: type[EmbedService] = EmbedService,
    tau: float = 0.90,
) -> None:
    """
    Sync implementation of anchor duplicate marking.
    """
    if not anchor_claim_id or not claim_verdicts:
        return
    if not embed_service.is_available():
        return

    anchor_norm = _norm_id(anchor_claim_id)
    anc = next(
        (
            cv
            for cv in claim_verdicts
            if isinstance(cv, dict) and _norm_id(cv.get("claim_id")) == anchor_norm
        ),
        None,
    )
    if not anc:
        return

    anchor_text = _claim_text(anc)
    if not anchor_text:
        return

    candidates: list[str] = []
    candidate_refs: list[dict] = []
    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        if _norm_id(cv.get("claim_id")) == anchor_norm:
            continue
        txt = _claim_text(cv)
        if not txt:
            continue
        candidates.append(txt)
        candidate_refs.append(cv)

    if not candidates:
        return

    scores = embed_service.batch_similarity(anchor_text, candidates)
    for cv, sim in zip(candidate_refs, scores):
        if not isinstance(sim, (int, float)) or not math.isfinite(float(sim)):
            continue
        if sim >= tau:
            cv["duplicate_of_anchor"] = True
            cv["dup_sim"] = float(sim)
        else:
            cv["duplicate_of_anchor"] = False


async def _mark_anchor_duplicates_async(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict],
    embed_service: type[EmbedService] = EmbedService,
    tau: float = 0.90,
) -> None:
    """
    Async version: runs embedding in thread pool to avoid blocking.
    """
    if not anchor_claim_id or not claim_verdicts:
        return
    if not embed_service.is_available():
        return
    
    import concurrent.futures
    from functools import partial
    
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(
            executor,
            partial(
                _mark_anchor_duplicates_sync,
                anchor_claim_id=anchor_claim_id,
                claim_verdicts=claim_verdicts,
                embed_service=embed_service,
                tau=tau,
            ),
        )


def _mark_anchor_duplicates(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict],
    embed_service: type[EmbedService] = EmbedService,
    tau: float = 0.90,
) -> None:
    """
    Mark secondary claims that are semantic duplicates of the anchor claim.
    Contract:
      - anchor vs secondary only
      - embedding-based cosine similarity
      - no filtering, only marking
    
    Note: This is a sync wrapper. Use _mark_anchor_duplicates_async in async context.
    """
    _mark_anchor_duplicates_sync(
        anchor_claim_id=anchor_claim_id,
        claim_verdicts=claim_verdicts,
        embed_service=embed_service,
        tau=tau,
    )

def _compute_article_g_from_anchor(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict] | None,
    prior_p: float = 0.5,
) -> tuple[float, dict]:
    """
    Pure formula (no orchestration heuristics):
      G = sigmoid( logit(prior_p) + k * logit(p_anchor) )
    where:
      p_anchor = verdict_score for anchor claim
      k = 0 if verdict_state is insufficient_evidence, else 1

    If anchor is missing or invalid, return prior_p (neutral).
    """
    debug = {
        "anchor_claim_id": anchor_claim_id,
        "used_anchor": False,
        "k": 0.0,
        "p_anchor": None,
        "prior_p": prior_p,
    }
    if not anchor_claim_id or not isinstance(claim_verdicts, list):
        return float(prior_p), debug

    anc_norm = _norm_id(anchor_claim_id)
    cv = next(
        (
            x
            for x in claim_verdicts
            if isinstance(x, dict) and _norm_id(x.get("claim_id")) == anc_norm
        ),
        None,
    )
    if not cv:
        return float(prior_p), debug

    p = cv.get("verdict_score")
    if not isinstance(p, (int, float)) or not math.isfinite(float(p)):
        return float(prior_p), debug
    p = float(p)
    if not (0.0 < p < 1.0):
        return float(prior_p), debug

    vs = str(cv.get("verdict_state") or "").strip().lower()
    k = 0.0 if vs == "insufficient_evidence" else 1.0
    debug.update({"used_anchor": True, "k": k, "p_anchor": p})

    L = _logit(float(prior_p)) + (k * _logit(p))
    return float(_sigmoid(L)), debug

ProgressCallback = Callable[[str], Awaitable[None]]


# Tier is a first-class prior on A (Explainability) ONLY.
# It must not bias veracity (G) or verdicts.
_TIER_A_PRIOR_MEAN = {
    "A": 0.96,
    "A'": 0.93,
    "B": 0.90,
    "C": 0.85,
    "D": 0.80,
    "UNKNOWN": 0.38,
}
_TIER_A_BASELINE = _TIER_A_PRIOR_MEAN["B"]


def _explainability_factor_for_tier(tier: str | None) -> tuple[float, str, float]:
    if not tier:
        prior = _TIER_A_PRIOR_MEAN["UNKNOWN"]
        return prior / _TIER_A_BASELINE, "unknown_default", prior
    prior = _TIER_A_PRIOR_MEAN.get(str(tier).strip().upper(), _TIER_A_PRIOR_MEAN["UNKNOWN"])
    return prior / _TIER_A_BASELINE, "best_tier", prior


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
    claim_extraction_text: str = ""


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
    claim_text_map: dict[str, str] = {}
    if claims:
        for c in claims:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("claim_id")
            if not cid:
                continue
            claim_text_map[str(cid)] = c.get("normalized_text") or c.get("text") or ""
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
        anchor_claim = pick_ui_main_claim(claims) or claims[0]
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
    if str(result.get("status", "")).lower() == "error":
        result["status"] = "error"
        Trace.event(
            "verdict.status",
            {"status": "error", "reason": "llm_error", "kind": "score_evidence"},
        )
    raw_verified_score = result.get("verified_score")
    raw_confidence_score = result.get("confidence_score")
    raw_rationale = result.get("rationale")
    Trace.event_full(
        "evidence.llm_output",
        {
            "verified_score_raw": raw_verified_score,
            "confidence_score_raw": raw_confidence_score,
            "rationale": raw_rationale,
            "claim_verdicts_raw": result.get("claim_verdicts"),
        },
    )

    # M93: Apply dependency penalties after scoring
    claim_verdicts = result.get("claim_verdicts")
    if isinstance(claim_verdicts, list):
        changed = apply_dependency_penalties(claim_verdicts, claims)
        # NOTE: dependency penalties mutate per-claim verdicts only.
        # Article-level G is selected later from the anchor claim,
        # and must NOT be recomputed as an average across claims.
        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            cid = cv.get("claim_id")
            if cid and "claim_text" not in cv:
                text_val = claim_text_map.get(str(cid))
                if text_val:
                    cv["claim_text"] = text_val

    conflict_detected = False
    verdict_state_by_claim: dict[str, str] = {}
    importance_by_claim: dict[str, float] = {}
    veracity_debug: list[dict] = []
    # NOTE: article-level G is computed from anchor formula (no worthiness/thesis fallback),
    # so we do not need claim_meta_by_id for G.
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        raw_id = claim.get("id") or claim.get("claim_id") or ""
        cid_norm = _norm_id(raw_id)
        if not cid_norm:
            continue
        try:
            importance_by_claim[cid_norm] = float(claim.get("importance", 1.0) or 1.0)
        except Exception:
            importance_by_claim[cid_norm] = 1.0
    
    if isinstance(claim_verdicts, list):
        # Prepare data for parallel processing
        def _process_single_cv(cv_data: dict) -> dict:
            """Process a single claim verdict - runs in thread pool."""
            cv = cv_data["cv"]
            claim_id = cv_data["claim_id"]
            claim_obj = cv_data["claim_obj"]
            temporality = cv_data["temporality"]
            has_direct_evidence = cv_data["has_direct_evidence"]
            pack_ref = cv_data["pack"]
            explainability = cv_data["explainability"]
            
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
                    pack_ref,
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
                    pack_ref,
                    policy={},
                    claim_id=claim_id,
                    temporality=temporality if isinstance(temporality, dict) else None,
                )
                cv["reasons_short"] = cv.get("reasons_short", []) or []

            # Explainability tier factor
            explainability_update = None
            if isinstance(explainability, (int, float)) and explainability >= 0:
                best_tier = agg.get("best_tier") if isinstance(agg, dict) else None
                pre_a = float(explainability)
                factor, source, prior = _explainability_factor_for_tier(best_tier)
                if math.isfinite(pre_a) and 0.0 < pre_a < 1.0:
                    if factor <= 0 or not math.isfinite(factor):
                        Trace.event(
                            "verdict.explainability_bad_factor",
                            {
                                "claim_id": claim_id,
                                "best_tier": best_tier,
                                "pre_A": pre_a,
                                "prior": prior,
                                "baseline": _TIER_A_BASELINE,
                                "factor": factor,
                                "source": source,
                            },
                        )
                        post_a = pre_a
                    else:
                        post_a = _sigmoid(_logit(pre_a) + math.log(factor))
                    if abs(post_a - pre_a) > 1e-9:
                        explainability_update = post_a
                    Trace.event(
                        "verdict.explainability_tier_factor",
                        {
                            "claim_id": claim_id,
                            "best_tier": best_tier,
                            "pre_A": pre_a,
                            "prior": prior,
                            "baseline": _TIER_A_BASELINE,
                            "factor": factor,
                            "post_A": post_a if explainability_update else pre_a,
                            "source": source,
                        },
                    )
                else:
                    Trace.event(
                        "verdict.explainability_missing",
                        {"claim_id": claim_id, "best_tier": best_tier, "pre_A": pre_a},
                    )

            veracity_entry = {
                "claim_id": claim_id,
                "role": claim_obj.get("claim_role") if claim_obj else None,
                "target": claim_obj.get("verification_target") if claim_obj else None,
                "harm": claim_obj.get("harm_potential") if claim_obj else None,
                "llm_verdict": cv.get("verdict"),
                "llm_score": cv.get("verdict_score"),
                "agg_verdict": agg.get("verdict") if isinstance(agg, dict) else cv.get("verdict"),
                "agg_score": agg.get("verdict_score") if isinstance(agg, dict) else cv.get("verdict_score"),
                "has_direct_evidence": has_direct_evidence,
                "best_tier": agg.get("best_tier") if isinstance(agg, dict) else None,
            }

            # Respect verdict_state if already normalized by parser
            existing_state = str(cv.get("verdict_state") or "").lower().strip()
            CANONICAL_STATES = {"supported", "refuted", "conflicted", "insufficient_evidence"}
            
            if existing_state in CANONICAL_STATES:
                # Already normalized by clamp_score_evidence_result()
                verdict_state = existing_state
            else:
                # Fallback mapping (should rarely happen after parser normalization)
                verdict_state = "insufficient_evidence"
                if cv["verdict"] == "verified":
                    verdict_state = "supported"
                elif cv["verdict"] == "refuted":
                    verdict_state = "refuted"
                elif cv["verdict"] == "ambiguous":
                    verdict_state = "conflicted"

            cv["verdict_state"] = verdict_state
            
            has_conflict = bool((cv.get("reasons_expert") or {}).get("conflict"))

            return {
                "claim_id": claim_id,
                "verdict_state": verdict_state,
                "veracity_entry": veracity_entry,
                "explainability_update": explainability_update,
                "has_conflict": has_conflict,
            }
        
        # Prepare all CV data
        cv_data_list = []
        items = pack.get("items", []) if isinstance(pack, dict) else []
        explainability = result.get("explainability_score", -1.0)
        
        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            claim_id = _norm_id(cv.get("claim_id"))
            if not claim_id and claims:
                claim_id = _norm_id(claims[0].get("id") or "c1")
                cv["claim_id"] = claim_id

            claim_obj = None
            for c in claims or []:
                if _norm_id(c.get("id") or c.get("claim_id")) == claim_id:
                    claim_obj = c
                    break
            temporality = claim_obj.get("temporality") if isinstance(claim_obj, dict) else None

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
            
            cv_data_list.append({
                "cv": cv,
                "claim_id": claim_id,
                "claim_obj": claim_obj,
                "temporality": temporality,
                "has_direct_evidence": has_direct_evidence,
                "pack": pack,
                "explainability": explainability,
            })
        
        # Process in parallel using thread pool
        import concurrent.futures
        
        if len(cv_data_list) > 1:
            # Parallel execution for multiple claims
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(cv_data_list))) as executor:
                results_list = list(executor.map(_process_single_cv, cv_data_list))
        else:
            # Sequential for single claim (avoid thread overhead)
            results_list = [_process_single_cv(d) for d in cv_data_list]
        
        # Merge results
        for res in results_list:
            verdict_state_by_claim[res["claim_id"]] = res["verdict_state"]
            veracity_debug.append(res["veracity_entry"])
            if res["explainability_update"] is not None:
                result["explainability_score"] = res["explainability_update"]
            if res["has_conflict"]:
                conflict_detected = True

        changed = apply_dependency_penalties(claim_verdicts, claims)
        # Do NOT recompute article-level G as an average across claims.
        # Article-level G is selected later from the anchor claim
        # and stored in result["verified_score"].
        if changed:
            result["dependency_penalty_applied"] = True

    # Anchor ↔ secondary semantic dedup (marking only)
    try:
        await _mark_anchor_duplicates_async(
            anchor_claim_id=anchor_claim_id,
            claim_verdicts=claim_verdicts if isinstance(claim_verdicts, list) else [],
            embed_service=EmbedService,
            tau=0.90,
        )
    except Exception as e:
        logger.warning("Anchor dedup failed (non-fatal): %s", e)

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
                    conf = cv.get("confidence")
                    if not _is_prob(conf):
                        conf = cv.get("verdict_score")
                    if not _is_prob(conf):
                        conf = 0.5
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
                cid = _norm_id(cv.get("claim_id"))
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
        belief_g = log_odds_to_prob(current_belief.log_odds)

        # Article-level G: pure anchor formula (no thesis/worthiness heuristics, no dedup).
        g_article, g_dbg = _compute_article_g_from_anchor(
            anchor_claim_id=anchor_claim_id,
            claim_verdicts=claim_verdicts if isinstance(claim_verdicts, list) else None,
            prior_p=0.5,
        )
        prev = result.get("verified_score")
        result["verified_score"] = g_article
        Trace.event(
            "verdict.article_g_formula",
            {
                **g_dbg,
                "prev_verified_score": prev,
                "belief_score": belief_g,
            },
        )

        Trace.event_full(
            "verdict.veracity_debug",
            {
                "raw_verified_score": raw_verified_score,
                "raw_confidence_score": raw_confidence_score,
                "final_verified_score": result.get("verified_score"),
                "final_confidence_score": result.get("confidence_score"),
                "veracity_by_claim": veracity_debug,
                "rationale": raw_rationale,
            },
        )
        
        # Trace
        result["bayesian_trace"] = {
            "prior_log_odds": inp.prior_belief.log_odds,
            "consensus_score": consensus.score,
            "posterior_log_odds": current_belief.log_odds,
            "final_probability": result["verified_score"],
            "belief_probability": belief_g,
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

    # Preserve stitched claim-extraction text for audit/history.
    if inp.claim_extraction_text:
        result["claim_extraction_text"] = inp.claim_extraction_text

    return result
