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
from typing import Any, Awaitable, Callable

import logging

from spectrue_core.utils.embedding_service import EmbedService
from spectrue_core.schema.signals import TimeWindow
from spectrue_core.schema.scoring import BeliefState
from spectrue_core.graph.context import ClaimContextGraph
# Bayesian scoring imports moved to bayesian_update.py (M119)
from spectrue_core.verification.temporal.temporal import (
    label_evidence_timeliness,
    normalize_time_window,
)
from spectrue_core.verification.search.source_utils import canonicalize_sources

# Suppress deprecation warning - full migration to Bayesian scoring is future work
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from spectrue_core.verification.scoring.rgba_aggregation import (
    apply_dependency_penalties,
    apply_conflict_explainability_penalty,
)
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry
from spectrue_core.verification.claims.claim_selection import pick_ui_main_claim
from spectrue_core.verification.evidence.evidence_pack import EvidencePack
from spectrue_core.verification.search.search_policy import (
    resolve_profile_name,
    resolve_stance_pass_mode,
)
from spectrue_core.utils.trace import Trace

# Extracted scoring helpers from previous refactoring
from spectrue_core.verification.evidence.evidence_scoring import (
    norm_id as _norm_id,
    mark_anchor_duplicates_async as _mark_anchor_duplicates_async,
)

# Explainability and stance processing modules
# Claim verdict processing
from spectrue_core.verification.evidence_verdict_processing import (
    process_claim_verdicts,
    enrich_all_claim_verdicts,
)

# Bayesian update logic (M119)
from spectrue_core.verification.evidence.bayesian_update import apply_bayesian_update


logger = logging.getLogger(__name__)


# NOTE: Helper functions (_norm_id, _is_prob, _logit, _sigmoid, _claim_text) 
# are now imported from evidence_scoring module (M118)



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



ProgressCallback = Callable[[str], Awaitable[None]]


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
    # Pipeline field removed - mode determined by score_mode parameter in run_evidence_flow()


@dataclass(frozen=True, slots=True)
class EvidenceCollection:
    """Output of evidence collection prior to judging."""

    pack: EvidencePack
    claims: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    claim_text_map: dict[str, str]
    anchor_claim: dict[str, Any] | None
    anchor_claim_id: str | None
    time_windows: dict[str, TimeWindow]
    current_cost: float


async def collect_evidence(
    *,
    agent,
    search_mgr,
    build_evidence_pack,
    calibration_registry: CalibrationRegistry | None = None,
    inp: EvidenceFlowInput,
    claims: list[dict],
    sources: list[dict],
) -> EvidenceCollection:
    """Collect and structure evidence without invoking the judge."""
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

    claim_text_map: dict[str, str] = {}
    if claims:
        for c in claims:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("claim_id")
            if not cid:
                continue
            claim_text_map[str(cid)] = c.get("normalized_text") or c.get("text") or ""

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

    # Language consistency validation (optional, just for tracing)
    if claims and inp.content_lang:
        from spectrue_core.utils.language_validation import (
            validate_claims_language_consistency,
        )
        lang_valid, lang_mismatches = validate_claims_language_consistency(
            claims, inp.content_lang, pipeline_mode="collect", min_confidence=0.7,
        )
        if not lang_valid:
            Trace.event("pipeline.language_mismatch_ignored", {
                "expected": inp.content_lang, "mismatches": lang_mismatches,
            })

    # Deterministic ranking
    claims.sort(key=lambda c: (-c.get("importance", 0.0), c.get("text", "")))

    pack = build_evidence_pack(
        fact=inp.original_fact,
        claims=claims,
        sources=sources,
        search_results_clustered=None,
        content_lang=inp.content_lang or inp.lang,
        article_context={"text_excerpt": inp.fact[:500]}
        if inp.fact != inp.original_fact
        else None,
    )

    return EvidenceCollection(
        pack=pack,
        claims=claims,
        sources=sources,
        claim_text_map=claim_text_map,
        anchor_claim=anchor_claim,
        anchor_claim_id=anchor_claim_id,
        time_windows=time_windows,
        current_cost=current_cost,
    )


async def annotate_evidence_stance(
    *,
    agent,
    inp: EvidenceFlowInput,
    claims: list[dict],
    sources: list[dict],
) -> list[dict]:
    """Optional stance annotation using clustering skill output."""
    if not claims or not sources:
        return []
    if inp.progress_callback:
        await inp.progress_callback("stance_annotation")
    profile_name = resolve_profile_name(inp.search_type)
    stance_pass_mode = resolve_stance_pass_mode(profile_name)
    return await agent.cluster_evidence(
        claims,
        sources,
        stance_pass_mode=stance_pass_mode,
    )


def rebuild_evidence_pack(
    *,
    build_evidence_pack,
    collection: EvidenceCollection,
    clustered_results: list[dict] | None,
    sources_override: list[dict] | None = None,
    inp: EvidenceFlowInput,
) -> EvidenceCollection:
    """Rebuild EvidencePack using optional clustered results."""
    sources = sources_override if sources_override is not None else collection.sources
    pack = build_evidence_pack(
        fact=inp.original_fact,
        claims=collection.claims,
        sources=sources,
        search_results_clustered=clustered_results,
        content_lang=inp.content_lang or inp.lang,
        article_context={"text_excerpt": inp.fact[:500]}
        if inp.fact != inp.original_fact
        else None,
    )

    return EvidenceCollection(
        pack=pack,
        claims=collection.claims,
        sources=sources,
        claim_text_map=collection.claim_text_map,
        anchor_claim=collection.anchor_claim,
        anchor_claim_id=collection.anchor_claim_id,
        time_windows=collection.time_windows,
        current_cost=collection.current_cost,
    )


async def score_evidence_collection(
    *,
    agent,
    search_mgr,
    enrich_sources_with_trust,
    inp: EvidenceFlowInput,
    collection: EvidenceCollection,
    score_mode: str = "standard",
) -> dict:
    """Invoke judge on a pre-collected evidence pack."""
    pack = collection.pack
    claims = collection.claims
    sources = collection.sources
    anchor_claim = collection.anchor_claim
    anchor_claim_id = collection.anchor_claim_id
    time_windows = collection.time_windows
    claim_text_map = collection.claim_text_map

    if inp.progress_callback:
        await inp.progress_callback("score_evidence")

    # LLM scoring call controlled by explicit score_mode parameter
    # - "parallel": per-claim scoring (deep mode)
    # - "standard": single LLM call for batch scoring (normal mode)
    use_parallel_scoring = score_mode == "parallel"

    if use_parallel_scoring:
        # Per-claim scoring: each claim gets its own LLM call with individual RGBA
        result = await agent.score_evidence_parallel(
            pack, model=inp.gpt_model, lang=inp.lang, max_concurrency=5
        )
    else:
        # Standard mode: single LLM call for batch scoring
        result = await agent.score_evidence(pack, model=inp.gpt_model, lang=inp.lang)

    # Track judge mode for downstream logic
    result["judge_mode"] = "deep" if use_parallel_scoring else "standard"

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
        # Process all claim verdicts using extracted function
        explainability_score = result.get("explainability_score", -1.0)

        verdict_state_by_claim, veracity_debug, conflict_detected, explainability_update = (
            process_claim_verdicts(
                claim_verdicts, claims, pack, explainability_score
            )
        )

        if explainability_update is not None:
            result["explainability_score"] = explainability_update

        # Prepare global RGBA for enrichment
        # IMPORTANT: These global_* values are ONLY used as fallback for standard mode.
        # In deep mode, each claim_verdict MUST have its own RGBA from claim-judge.
        global_r = float(result.get("danger_score", -1.0))
        if global_r < 0:
            global_r = 0.0

        global_b = float(result.get("style_score", -1.0))
        if global_b < 0:
            global_b = float(result.get("context_score", -1.0))
        if global_b < 0:
            global_b = 1.0  # Default B=1.0 if missing

        global_a = float(result.get("explainability_score", -1.0))
        if global_a < 0:
            global_a = 1.0  # Default A=1.0 if missing

        # Enrich all claim verdicts with sources and RGBA
        enrich_all_claim_verdicts(
            claim_verdicts,
            pack,
            enrich_sources_with_trust,
            (global_r, 0.0, global_b, global_a),  # G will be set per-claim
            result.get("judge_mode", "standard"),
        )

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

    # Bayesian Scoring (extracted to bayesian_update.py)
    if inp.prior_belief:
        result["bayesian_trace"] = apply_bayesian_update(
            prior_belief=inp.prior_belief,
            context_graph=inp.context_graph,
            claim_verdicts=claim_verdicts if isinstance(claim_verdicts, list) else [],
            anchor_claim_id=anchor_claim_id,
            importance_by_claim=importance_by_claim,
            veracity_debug=veracity_debug,
            pack=pack,
            result=result,
            raw_verified_score=raw_verified_score,
            raw_confidence_score=raw_confidence_score,
            raw_rationale=raw_rationale,
        )

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

    result["cost"] = collection.current_cost
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
    # -1.0 is now a valid value meaning "unverified/unknown" - do not override
    if verified is None:
        logger.warning(
            "[Pipeline] ⚠️ Missing verified_score in result - using -1.0 (unverified)"
        )
        result["verified_score"] = -1.0

    # Preserve stitched claim-extraction text for audit/history.
    if inp.claim_extraction_text:
        result["claim_extraction_text"] = inp.claim_extraction_text

    # Add baseline pricing policy snapshot for audit
    result["aggregation_policy"] = _aggregation_policy(search_mgr)

    return result


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
    score_mode: str = "standard",  # "standard" (single LLM call) or "parallel" (per-claim)
    cluster_evidence: bool = True,
) -> dict:
    """
    Analysis + scoring: clustering, social verification, evidence pack, scoring, finalize.

    Args:
        score_mode: Scoring strategy - "standard" for single LLM call (normal mode),
                   "parallel" for per-claim scoring (deep mode).

    Matches existing pipeline behavior; expects sources/claims to be mutable dict shapes.
    """

    collection = await collect_evidence(
        agent=agent,
        search_mgr=search_mgr,
        build_evidence_pack=build_evidence_pack,
        calibration_registry=calibration_registry,
        inp=inp,
        claims=claims,
        sources=sources,
    )

    if cluster_evidence:
        clustered = await annotate_evidence_stance(
            agent=agent,
            inp=inp,
            claims=claims,
            sources=collection.sources,
        )
        if clustered:
            collection = rebuild_evidence_pack(
                build_evidence_pack=build_evidence_pack,
                collection=collection,
                clustered_results=clustered,
                sources_override=clustered,
                inp=inp,
            )

    return await score_evidence_collection(
        agent=agent,
        search_mgr=search_mgr,
        enrich_sources_with_trust=enrich_sources_with_trust,
        inp=inp,
        collection=collection,
        score_mode=score_mode,
    )
