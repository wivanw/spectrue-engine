# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

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
from spectrue_core.graph.propagation import (
    propagate_belief,
    propagation_routing_signals,
)
from spectrue_core.scoring.belief import (
    calculate_evidence_impact,
    update_belief,
    log_odds_to_prob,
    apply_consensus_bound,
)
from spectrue_core.scoring.consensus import calculate_consensus
from spectrue_core.scoring.claim_posterior import (
    compute_claim_posterior,
    aggregate_article_posterior,
    EvidenceItem,
    PosteriorParams,
)
from spectrue_core.verification.temporal import (
    label_evidence_timeliness,
    normalize_time_window,
)
from spectrue_core.verification.source_utils import canonicalize_sources

# Suppress deprecation warning - full migration to Bayesian scoring is future work
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from spectrue_core.verification.scoring_aggregation import aggregate_claim_verdict

from spectrue_core.verification.rgba_aggregation import (
    apply_dependency_penalties,
    apply_conflict_explainability_penalty,
)
from spectrue_core.verification.calibration_registry import CalibrationRegistry
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
    return (
        isinstance(x, (int, float))
        and math.isfinite(float(x))
        and 0.0 <= float(x) <= 1.0
    )


def _logit(p: float) -> float:
    # Contract: p must be in (0,1). No clamping here.
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _claim_text(cv: dict) -> str:
    text = cv.get("claim_text") or cv.get("claim") or cv.get("text") or ""
    return str(text).strip()


def _aggregation_policy(search_mgr) -> dict:
    calibration = getattr(
        getattr(getattr(search_mgr, "config", None), "runtime", None),
        "calibration",
        None,
    )
    if not calibration:
        return {}
    return {
        "penalty_conflict_weight": float(calibration.penalty_conflict_weight),
        "penalty_temporal_weight": float(calibration.penalty_temporal_weight),
        "penalty_diversity_weight": float(calibration.penalty_diversity_weight),
    }


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
    prior = _TIER_A_PRIOR_MEAN.get(
        str(tier).strip().upper(), _TIER_A_PRIOR_MEAN["UNKNOWN"]
    )
    return prior / _TIER_A_BASELINE, "best_tier", prior


def _select_anchor_for_article_g(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict] | None,
    veracity_debug: list[dict],
) -> tuple[str | None, dict]:
    debug: dict = {
        "pre_anchor_id": anchor_claim_id,
        "selected_anchor_id": anchor_claim_id,
        "override_used": False,
        "reason": "no_candidates",
    }
    if not claim_verdicts or not veracity_debug:
        return anchor_claim_id, debug

    verdict_score_by_claim: dict[str, float] = {}
    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        cid = _norm_id(cv.get("claim_id"))
        if not cid:
            continue
        score = cv.get("verdict_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            verdict_score_by_claim[cid] = float(score)

    candidates: list[dict] = []
    for entry in veracity_debug:
        if not isinstance(entry, dict):
            continue
        cid = _norm_id(entry.get("claim_id"))
        if not cid:
            continue
        if not entry.get("has_direct_evidence"):
            continue
        p = verdict_score_by_claim.get(cid)
        if p is None or not (0.0 < p < 1.0):
            continue
        best_tier = entry.get("best_tier")
        factor, source, _prior = _explainability_factor_for_tier(best_tier)
        if not isinstance(factor, (int, float)) or not math.isfinite(float(factor)):
            continue
        factor = max(0.0, float(factor))
        score = abs(p - 0.5) * factor
        candidates.append(
            {
                "id": cid,
                "score": score,
                "p": p,
                "best_tier": best_tier,
                "factor": factor,
                "factor_source": source,
            }
        )

    if not candidates:
        return anchor_claim_id, debug

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    if best["score"] <= 0:
        debug["reason"] = "non_informative"
        return anchor_claim_id, debug

    selected_id = best["id"]
    debug.update(
        {
            "selected_anchor_id": selected_id,
            "override_used": _norm_id(selected_id) != _norm_id(anchor_claim_id),
            "reason": "evidence_weighted_distance",
            "selected_score": best["score"],
            "selected_p": best["p"],
            "selected_best_tier": best["best_tier"],
            "selected_factor": best["factor"],
            "selected_factor_source": best["factor_source"],
            "candidate_count": len(candidates),
            "top_candidates": candidates[:3],
        }
    )
    return selected_id, debug


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
    # M113: Pipeline profile name (e.g., 'normal', 'deep').
    pipeline: str | None = None


async def run_evidence_flow(
    *,
    agent,
    search_mgr,
    build_evidence_pack,
    enrich_sources_with_trust,
    calibration_registry: CalibrationRegistry | None = None,
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
    agg_policy = _aggregation_policy(search_mgr)

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
        anchor_claim = (
            pick_ui_main_claim(
                claims,
                calibration_registry=calibration_registry,
            )
            or claims[0]
        )
        anchor_claim_id = anchor_claim.get("id") or anchor_claim.get("claim_id")

    # NORMAL pipeline must be SINGLE-CLAIM (execution unit invariant).
    # This prevents multi-claim batch leakage (c1/c2 mentions) and cross-language mixing.
    # NOTE: pipeline profile is passed via inp.pipeline by ValidationPipeline.
    pipeline_profile = inp.pipeline or "normal"

    # Language consistency validation (Phase 4 invariant)
    if claims and inp.content_lang:
        from spectrue_core.utils.language_validation import (
            validate_claims_language_consistency,
        )

        lang_valid, lang_mismatches = validate_claims_language_consistency(
            claims,
            inp.content_lang,
            pipeline_mode=pipeline_profile,
            min_confidence=0.7,
        )
        if not lang_valid and pipeline_profile == "normal":
            # In normal mode, language mismatch is a violation
            raise RuntimeError(
                f"Language mismatch in normal pipeline: expected={inp.content_lang}, "
                f"mismatches={lang_mismatches}"
            )

    if pipeline_profile == "normal" and claims:
        if anchor_claim_id:
            claims = [
                c
                for c in claims
                if isinstance(c, dict)
                and str(c.get("id") or c.get("claim_id")) == str(anchor_claim_id)
            ]
        if len(claims) != 1:
            raise RuntimeError(
                f"Normal evidence flow violation: expected 1 claim, got {len(claims)}"
            )
        Trace.event(
            "evidence_flow.single_claim.enforced",
            {
                "profile": pipeline_profile,
                "anchor_claim_id": str(anchor_claim_id or ""),
            },
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

    # LLM call.
    # - Language is explicitly fixed by inp.lang.
    # - In 'normal' profile, `claims` is enforced to exactly one claim.
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

    # Apply dependency penalties after scoring
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
        # Helper to get tier rank
        def _tier_rank(tier: str | None) -> int:
            if not tier:
                return 0
            return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(
                str(tier).strip().upper(), 0
            )

        # Prepare data for parallel processing
        def _process_single_cv(cv_data: dict) -> dict:
            """Process a single claim verdict using unified Bayesian posterior."""
            cv = cv_data["cv"]
            claim_id = cv_data["claim_id"]
            claim_obj = cv_data["claim_obj"]
            temporality = cv_data["temporality"]
            has_direct_evidence = cv_data["has_direct_evidence"]
            pack_ref = cv_data["pack"]
            explainability = cv_data["explainability"]

            # Get LLM verdict score (raw observation)
            llm_score = cv.get("verdict_score")
            if not isinstance(llm_score, (int, float)):
                llm_score = 0.5
            llm_score = float(llm_score)

            # Build evidence items from pack
            evidence_items = []
            items = pack_ref.get("items", []) if isinstance(pack_ref, dict) else []
            best_tier = None

            for item in items:
                if not isinstance(item, dict):
                    continue
                item_claim_id = item.get("claim_id")
                if claim_id and item_claim_id not in (None, claim_id):
                    continue

                stance = str(item.get("stance") or "").lower()
                tier = item.get("tier")
                relevance = item.get("relevance")
                if not isinstance(relevance, (int, float)):
                    relevance = 0.5
                quote_present = bool(item.get("quote"))

                evidence_items.append(
                    EvidenceItem(
                        stance=stance,
                        tier=tier,
                        relevance=float(relevance),
                        quote_present=quote_present,
                        claim_id=claim_id,
                    )
                )

                # Track best tier
                if tier and (
                    best_tier is None or _tier_rank(tier) > _tier_rank(best_tier)
                ):
                    best_tier = tier

            # Compute unified posterior
            posterior_result = compute_claim_posterior(
                llm_verdict_score=llm_score,
                best_tier=best_tier,
                evidence_items=evidence_items,
                claim_id=claim_id,
            )

            # Update cv with posterior score
            cv["verdict_score"] = posterior_result.p_posterior
            cv["posterior_result"] = posterior_result.to_dict()

            # Derive verdict from posterior
            if posterior_result.p_posterior > 0.65:
                cv["verdict"] = "verified"
                cv["status"] = "verified"
            elif posterior_result.p_posterior < 0.35:
                cv["verdict"] = "refuted"
                cv["status"] = "refuted"
            else:
                cv["verdict"] = "ambiguous"
                cv["status"] = "ambiguous"

            # Legacy reasons_expert for backward compatibility
            cv["reasons_expert"] = {
                "best_tier": best_tier,
                "n_support": posterior_result.n_support,
                "n_refute": posterior_result.n_refute,
                "effective_support": posterior_result.effective_support,
                "effective_refute": posterior_result.effective_refute,
            }
            cv["reasons_short"] = cv.get("reasons_short", []) or []

            # Explainability tier factor
            explainability_update = None
            if isinstance(explainability, (int, float)) and explainability >= 0:
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
                "llm_score": llm_score,  # Original LLM score
                "posterior_score": posterior_result.p_posterior,
                "has_direct_evidence": has_direct_evidence,
                "best_tier": best_tier,
            }

            # Respect verdict_state if already normalized by parser
            existing_state = str(cv.get("verdict_state") or "").lower().strip()
            CANONICAL_STATES = {
                "supported",
                "refuted",
                "conflicted",
                "insufficient_evidence",
            }

            if existing_state in CANONICAL_STATES:
                verdict_state = existing_state
            else:
                # Derive from posterior
                verdict_state = "insufficient_evidence"
                if posterior_result.p_posterior > 0.65:
                    verdict_state = "supported"
                elif posterior_result.p_posterior < 0.35:
                    verdict_state = "refuted"
                elif posterior_result.n_support > 0 or posterior_result.n_refute > 0:
                    verdict_state = "conflicted"

            cv["verdict_state"] = verdict_state

            # Conflict detection from evidence balance
            has_conflict = (
                posterior_result.n_support > 0 and posterior_result.n_refute > 0
            )

            return {
                "claim_id": claim_id,
                "verdict_state": verdict_state,
                "veracity_entry": veracity_entry,
                "explainability_update": explainability_update,
                "has_conflict": has_conflict,
                "posterior_score": posterior_result.p_posterior,
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
            temporality = (
                claim_obj.get("temporality") if isinstance(claim_obj, dict) else None
            )

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

            cv_data_list.append(
                {
                    "cv": cv,
                    "claim_id": claim_id,
                    "claim_obj": claim_obj,
                    "temporality": temporality,
                    "has_direct_evidence": has_direct_evidence,
                    "pack": pack,
                    "explainability": explainability,
                }
            )

        # Process in parallel using thread pool
        import concurrent.futures

        if len(cv_data_list) > 1:
            # Parallel execution for multiple claims
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, len(cv_data_list))
            ) as executor:
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

        # Enrich verdicts with per-claim sources and RGBA
        # This fixes the UI showing cumulative sources and identical RGBA for all claims
        # RGBA order: [R=danger, G=verified, B=style, A=explainability]
        global_r = float(result.get("danger_score", -1.0))
        if global_r < 0:
            global_r = 0.0

        global_b = float(result.get("style_score", -1.0))
        if global_b < 0:
            global_b = float(result.get("context_score", -1.0))
        if global_b < 0:
            global_b = 1.0  # Default B=1.0 if missing?

        global_a = float(result.get("explainability_score", -1.0))
        if global_a < 0:
            global_a = 1.0  # Default A=1.0 if missing?

        all_scored = pack.get("scored_sources") or []
        all_context = pack.get("context_sources") or []

        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            cid = _norm_id(cv.get("claim_id"))
            if not cid:
                continue

            # Filter sources for this claim
            # Include: sources matching this claim_id OR shared sources (claim_id=None)
            claim_sources = []
            seen_urls = set()
            for s in all_scored + all_context:
                if not isinstance(s, dict):
                    continue
                scid = _norm_id(s.get("claim_id"))
                # Match claim-specific sources OR shared sources (no claim_id)
                if scid == cid or scid is None or s.get("claim_id") is None:
                    url = s.get("url")
                    if url and url not in seen_urls:
                        claim_sources.append(s)
                        seen_urls.add(url)

            # Enrich per-claim sources
            cv["sources"] = enrich_sources_with_trust(claim_sources)

            # Calculate per-claim RGBA
            # G is specific to claim
            g_score = float(cv.get("verdict_score", 0.5) or 0.5)

            # R, B, A are global for now (unless claim has specific overrides later)
            cv["rgba"] = [global_r, g_score, global_b, global_a]

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

    # Bayesian Scoring
    if inp.prior_belief:
        current_belief = inp.prior_belief

        # Consensus Calculation
        evidence_list = []
        raw_evidence = (
            getattr(pack, "evidence", [])
            if not isinstance(pack, dict)
            else pack.get("evidence", [])
        )

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
                    current_belief = update_belief(
                        current_belief, anchor_node.propagated_belief.log_odds
                    )

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
                impact = calculate_evidence_impact(
                    v, confidence=strength, relevance=relevance
                )
                current_belief = update_belief(current_belief, impact)

        # Apply Consensus
        current_belief = apply_consensus_bound(current_belief, consensus)
        belief_g = log_odds_to_prob(current_belief.log_odds)

        anchor_for_g = anchor_claim_id
        anchor_dbg = {}
        if isinstance(claim_verdicts, list) and veracity_debug:
            anchor_for_g, anchor_dbg = _select_anchor_for_article_g(
                anchor_claim_id=anchor_claim_id,
                claim_verdicts=claim_verdicts,
                veracity_debug=veracity_debug,
            )
            Trace.event("anchor_selection.post_evidence", anchor_dbg)

        # Article-level G: pure anchor formula (no thesis/worthiness heuristics, no dedup).
        g_article, g_dbg = _compute_article_g_from_anchor(
            anchor_claim_id=anchor_for_g,
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
            timeliness_labels.append({"source_url": url, "timeliness_status": status})
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
