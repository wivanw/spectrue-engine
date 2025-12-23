from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import asyncio
import logging

from spectrue_core.verification.source_utils import canonicalize_sources
from spectrue_core.verification.trusted_sources import is_social_platform
from spectrue_core.verification.rgba_aggregation import (
    apply_dependency_penalties,
    recompute_verified_score,
)
from spectrue_core.verification.search_policy import (
    resolve_profile_name,
    resolve_stance_pass_mode,
)

logger = logging.getLogger(__name__)

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
    if claims:
        core_claims = [c for c in claims if c.get("type") == "core"]
        anchor_claim = (
            max(core_claims, key=lambda c: c.get("importance", 0))
            if core_claims
            else claims[0]
        )

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

    verified = result.get("verified_score", -1.0)
    if verified < 0:
        logger.warning("[Pipeline] ⚠️ Missing verified_score in result - using 0.5")
        verified = 0.5
        result["verified_score"] = verified

    return result
