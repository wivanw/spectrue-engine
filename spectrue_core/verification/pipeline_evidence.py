from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import asyncio
import logging

from spectrue_core.verification.source_utils import canonicalize_sources
from spectrue_core.verification.trusted_sources import is_social_platform
from spectrue_core.verification.evidence import (
    is_strong_tier,
    strongest_tiers_by_claim,
)
from spectrue_core.verification.rgba_aggregation import (
    apply_dependency_penalties,
    apply_conflict_explainability_penalty,
    recompute_verified_score,
)
from spectrue_core.verification.search_policy import (
    resolve_profile_name,
    resolve_stance_pass_mode,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]

REFUTE_CAP = 0.2
SUPPORT_FLOOR = 0.8
CONFLICT_MIN = 0.35
CONFLICT_MAX = 0.65


def _state_from_score(score: float) -> str:
    if score <= REFUTE_CAP:
        return "refuted"
    if score >= SUPPORT_FLOOR:
        return "supported"
    return "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class EvidenceFlowInput:
    fact: str
    original_fact: str
    lang: str
    content_lang: str | None
    gpt_model: str
    search_type: str
    progress_callback: ProgressCallback | None


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
                    logger.info(
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
    if isinstance(claim_verdicts, list):
        tier_summary = strongest_tiers_by_claim(pack.get("scored_sources", []))
        for cv in claim_verdicts:
            if not isinstance(cv, dict):
                continue
            claim_id = str(cv.get("claim_id") or "").strip()
            if not claim_id and claims:
                claim_id = str(claims[0].get("id") or "c1")
                cv["claim_id"] = claim_id

            tiers = tier_summary.get(claim_id, {})
            support_tier = tiers.get("support_tier")
            refute_tier = tiers.get("refute_tier")

            score = float(cv.get("verdict_score", 0.5) or 0.5)

            if support_tier is None and refute_tier is None:
                verdict_state = _state_from_score(score)
                cv["verdict_state"] = verdict_state
                verdict_state_by_claim[claim_id] = verdict_state
                continue

            support_strong = is_strong_tier(support_tier)
            refute_strong = is_strong_tier(refute_tier)

            if support_strong and refute_strong:
                conflict_detected = True
                score = min(max(score, CONFLICT_MIN), CONFLICT_MAX)
                cv["verdict_score"] = score
                verdict_state = "conflicted"
            elif refute_strong:
                cv["verdict_score"] = min(score, REFUTE_CAP)
                verdict_state = "refuted"
            elif support_strong:
                cv["verdict_score"] = max(score, SUPPORT_FLOOR)
                verdict_state = "supported"
            else:
                verdict_state = "insufficient_evidence"

            cv["verdict_state"] = verdict_state
            verdict_state_by_claim[claim_id] = verdict_state

        recalculated = recompute_verified_score(claim_verdicts)
        if recalculated is not None:
            result["verified_score"] = recalculated

    if conflict_detected:
        explainability = result.get("explainability_score", -1.0)
        if isinstance(explainability, (int, float)) and explainability >= 0:
            result["explainability_score"] = apply_conflict_explainability_penalty(
                float(explainability),
            )

    if inp.progress_callback:
        await inp.progress_callback("finalizing")

    result["cost"] = current_cost
    result["text"] = inp.fact
    result["search_meta"] = search_mgr.get_search_meta()
    result["sources"] = enrich_sources_with_trust(sources)

    if anchor_claim:
        result["anchor_claim"] = {
            "text": anchor_claim.get("text", ""),
            "type": anchor_claim.get("type", "core"),
            "importance": anchor_claim.get("importance", 1.0),
        }
        if anchor_claim_id and anchor_claim_id in verdict_state_by_claim:
            result["verdict_state"] = verdict_state_by_claim[anchor_claim_id]

    verified = result.get("verified_score", -1.0)
    if verified < 0:
        logger.warning("[Pipeline] ⚠️ Missing verified_score in result - using 0.5")
        verified = 0.5
        result["verified_score"] = verified

    return result
