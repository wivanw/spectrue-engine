"""Claim extraction use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.utils.trace import Trace
from spectrue_core.utils.coverage_anchors import extract_all_anchors
from spectrue_core.verification.claims.claim_dedup import dedup_claims_post_extraction_async


async def extract_claims_from_text(*, agent: Any, fact: str, lang: str, anchors: list[str] | None):
    result_tuple = await agent.extract_claims(
        text=fact,
        lang=lang,
        anchors=anchors,
    )
    claims, check_oracle, intent, fast_query = result_tuple
    if not claims:
        claims = [{"id": "c1", "text": fact[:500], "importance": 1.0}]
    return claims, check_oracle, intent, fast_query


async def dedup_claims_after_extraction(claims: list[dict]) -> tuple[list[dict], list[Any]]:
    before_n = len(claims)
    try:
        claims, dedup_pairs = await dedup_claims_post_extraction_async(claims, tau=0.90)
        after_n = len(claims)
        if dedup_pairs:
            Trace.event(
                "claims.dedup_post_extraction",
                {
                    "before": before_n,
                    "after": after_n,
                    "removed": max(before_n - after_n, 0),
                    "tau": 0.90,
                    "pairs": [
                        {
                            "canonical_id": p.canonical_id,
                            "duplicate_id": p.duplicate_id,
                            "sim": p.similarity,
                        }
                        for p in dedup_pairs[:50]
                    ],
                },
            )
        else:
            Trace.event(
                "claims.dedup_post_extraction",
                {"before": before_n, "after": after_n, "removed": 0, "tau": 0.90, "pairs": []},
            )
        return claims, dedup_pairs
    except Exception as exc:
        Trace.event("claims.dedup_post_extraction.failed", {"error": str(exc)})
        return claims, []


def ensure_anchors(*, fact: str, cached_anchors: list[str] | None):
    if cached_anchors is None:
        return extract_all_anchors(fact)
    return cached_anchors
