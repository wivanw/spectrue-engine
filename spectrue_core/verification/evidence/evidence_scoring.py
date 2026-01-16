# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Evidence Scoring

Bayesian scoring, anchor selection, and verdict computation utilities.
Extracted from pipeline_evidence.py for better modularity.

Part of pipeline decomposition refactor.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from spectrue_core.utils.trace import Trace

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# --- Math Helpers ---

def norm_id(x: Any) -> str:
    """Normalize ID to lowercase string."""
    return str(x or "").strip().lower()


def is_prob(x: Any) -> bool:
    """Check if value is a valid probability in [0, 1]."""
    return (
        isinstance(x, (int, float))
        and math.isfinite(float(x))
        and 0.0 <= float(x) <= 1.0
    )


def logit(p: float) -> float:
    """Compute logit (log-odds) of probability p."""
    # Contract: p must be in (0,1). No clamping here.
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Compute sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x))


def claim_text(cv: dict) -> str:
    """Extract claim text from verdict dict."""
    text = cv.get("claim_text") or cv.get("claim") or cv.get("text") or ""
    return str(text).strip()


# --- Tier-Based Explainability ---

# Tier is a first-class prior on A (Explainability) ONLY.
# It must not bias veracity (G) or verdicts.
TIER_A_PRIOR_MEAN = {
    "A": 0.96,
    "A'": 0.93,
    "B": 0.90,
    "C": 0.85,
    "D": 0.80,
    "UNKNOWN": 0.38,
}
TIER_A_BASELINE = TIER_A_PRIOR_MEAN["B"]


def explainability_factor_for_tier(tier: str | None) -> tuple[float, str, float]:
    """
    Compute explainability adjustment factor based on evidence tier.

    Args:
        tier: Evidence tier (A, A', B, C, D) or None

    Returns:
        Tuple of (factor, source_label, prior)
    """
    if not tier:
        prior = TIER_A_PRIOR_MEAN["UNKNOWN"]
        return prior / TIER_A_BASELINE, "unknown_default", prior
    prior = TIER_A_PRIOR_MEAN.get(
        str(tier).strip().upper(), TIER_A_PRIOR_MEAN["UNKNOWN"]
    )
    return prior / TIER_A_BASELINE, "best_tier", prior


def tier_rank(tier: str | None) -> int:
    """Get numeric rank for tier comparison."""
    if not tier:
        return 0
    return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(
        str(tier).strip().upper(), 0
    )


# --- Article G Computation ---

def compute_article_g_from_anchor(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict] | None,
    prior_p: float = 0.5,
) -> tuple[float, dict]:
    """
    Compute article-level G (veracity) from anchor claim.

    Pure formula (no orchestration heuristics):
      G = sigmoid( logit(prior_p) + k * logit(p_anchor) )
    where:
      p_anchor = verdict_score for anchor claim
      k = 0 if verdict_state is insufficient_evidence, else 1

    If anchor is missing or invalid, return prior_p (neutral).

    Args:
        anchor_claim_id: ID of the anchor claim
        claim_verdicts: List of claim verdict dicts
        prior_p: Prior probability (default 0.5)

    Returns:
        Tuple of (G score, debug dict)
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

    anc_norm = norm_id(anchor_claim_id)
    cv = next(
        (
            x
            for x in claim_verdicts
            if isinstance(x, dict) and norm_id(x.get("claim_id")) == anc_norm
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

    L = logit(float(prior_p)) + (k * logit(p))
    return float(sigmoid(L)), debug


def select_anchor_for_article_g(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict] | None,
    veracity_debug: list[dict],
) -> tuple[str | None, dict]:
    """
    Select best anchor claim for article G computation.

    Selects the claim with highest evidence-weighted distance from 0.5.

    Args:
        anchor_claim_id: Pre-selected anchor ID (may be overridden)
        claim_verdicts: List of claim verdict dicts
        veracity_debug: List of veracity debug entries

    Returns:
        Tuple of (selected anchor ID, debug dict)
    """
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
        cid = norm_id(cv.get("claim_id"))
        if not cid:
            continue
        score = cv.get("verdict_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            verdict_score_by_claim[cid] = float(score)

    candidates: list[dict] = []
    for entry in veracity_debug:
        if not isinstance(entry, dict):
            continue
        cid = norm_id(entry.get("claim_id"))
        if not cid:
            continue
        if not entry.get("has_direct_evidence"):
            continue
        p = verdict_score_by_claim.get(cid)
        if p is None or not (0.0 < p < 1.0):
            continue
        best_tier = entry.get("best_tier")
        factor, source, _prior = explainability_factor_for_tier(best_tier)
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
            "override_used": norm_id(selected_id) != norm_id(anchor_claim_id),
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


# --- Anchor Duplicate Marking ---

def mark_anchor_duplicates_sync(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict],
    embed_service: type,  # EmbedService type
    tau: float = 0.90,
) -> None:
    """
    Mark secondary claims that are semantic duplicates of the anchor claim.

    Sync implementation.

    Args:
        anchor_claim_id: ID of the anchor claim
        claim_verdicts: List of claim verdict dicts (mutated in place)
        embed_service: EmbedService class for embeddings
        tau: Similarity threshold (default 0.90)
    """
    if not anchor_claim_id or not claim_verdicts:
        return
    if not embed_service.is_available():
        return

    anchor_norm = norm_id(anchor_claim_id)
    anc = next(
        (
            cv
            for cv in claim_verdicts
            if isinstance(cv, dict) and norm_id(cv.get("claim_id")) == anchor_norm
        ),
        None,
    )
    if not anc:
        return

    anchor_text = claim_text(anc)
    if not anchor_text:
        return

    candidates: list[str] = []
    candidate_refs: list[dict] = []
    for cv in claim_verdicts:
        if not isinstance(cv, dict):
            continue
        if norm_id(cv.get("claim_id")) == anchor_norm:
            continue
        txt = claim_text(cv)
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


async def mark_anchor_duplicates_async(
    *,
    anchor_claim_id: str | None,
    claim_verdicts: list[dict],
    embed_service: type,  # EmbedService type
    tau: float = 0.90,
) -> None:
    """
    Async version: runs embedding in thread pool to avoid blocking.
    """
    if not anchor_claim_id or not claim_verdicts:
        return
    if not embed_service.is_available():
        return

    import asyncio
    import concurrent.futures
    from functools import partial

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(
            executor,
            partial(
                mark_anchor_duplicates_sync,
                anchor_claim_id=anchor_claim_id,
                claim_verdicts=claim_verdicts,
                embed_service=embed_service,
                tau=tau,
            ),
        )


# --- Verdict Processing ---

def derive_verdict_from_score(llm_score: float) -> tuple[str, str]:
    """
    Derive verdict and status from LLM score.

    Args:
        llm_score: Raw LLM score (0-1)

    Returns:
        Tuple of (verdict, status)
    """
    if llm_score > 0.65:
        return "verified", "verified"
    elif llm_score < 0.35:
        return "refuted", "refuted"
    else:
        return "ambiguous", "ambiguous"


def derive_verdict_state(
    llm_score: float,
    n_support: int,
    n_refute: int,
    existing_state: str | None = None,
) -> str:
    """
    Derive verdict state from LLM score and evidence counts.

    Args:
        llm_score: Raw LLM score (0-1)
        n_support: Number of supporting evidence items
        n_refute: Number of refuting evidence items
        existing_state: Existing state if already normalized

    Returns:
        Canonical verdict state string
    """
    CANONICAL_STATES = {
        "supported",
        "refuted",
        "conflicted",
        "insufficient_evidence",
    }

    if existing_state and existing_state.lower().strip() in CANONICAL_STATES:
        return existing_state.lower().strip()

    if llm_score > 0.65:
        return "supported"
    elif llm_score < 0.35:
        return "refuted"
    elif n_support > 0 or n_refute > 0:
        return "conflicted"
    else:
        return "insufficient_evidence"


def apply_explainability_tier_factor(
    pre_a: float,
    best_tier: str | None,
    claim_id: str | None = None,
) -> float | None:
    """
    Apply tier-based explainability factor to A score.

    Args:
        pre_a: Pre-factor A score (0-1)
        best_tier: Best evidence tier
        claim_id: Claim ID for tracing

    Returns:
        Updated A score, or None if no update needed
    """
    if not (isinstance(pre_a, (int, float)) and 0.0 < pre_a < 1.0):
        Trace.event(
            "verdict.explainability_missing",
            {"claim_id": claim_id, "best_tier": best_tier, "pre_A": pre_a},
        )
        return None

    factor, source, prior = explainability_factor_for_tier(best_tier)
    if factor <= 0 or not math.isfinite(factor):
        Trace.event(
            "verdict.explainability_bad_factor",
            {
                "claim_id": claim_id,
                "best_tier": best_tier,
                "pre_A": pre_a,
                "prior": prior,
                "baseline": TIER_A_BASELINE,
                "factor": factor,
                "source": source,
            },
        )
        return None

    post_a = sigmoid(logit(pre_a) + math.log(factor))
    if abs(post_a - pre_a) > 1e-9:
        Trace.event(
            "verdict.explainability_tier_factor",
            {
                "claim_id": claim_id,
                "best_tier": best_tier,
                "pre_A": pre_a,
                "prior": prior,
                "baseline": TIER_A_BASELINE,
                "factor": factor,
                "post_A": post_a,
                "source": source,
            },
        )
        return post_a

    return None
