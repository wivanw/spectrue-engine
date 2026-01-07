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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.contracts import CLAIMS_KEY, ClaimItem, Claims
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ExtractClaimsStep:
    """
    Extract verifiable claims from input text.

    Uses LLM to decompose text into atomic claims with metadata.
    Selects anchor claim for normal mode.

    Context Input:
        - extras: prepared_fact
        - lang

    Context Output:
        - claims (updated with extracted claims)
        - extras: anchor_claim_id, eligible_claims
    """

    agent: Any  # FactCheckerAgent
    stage: str = "retrieval_planning"
    name: str = "extract_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Extract claims from fact."""
        try:
            if self.stage == "post_evidence":
                claims = ctx.claims or []
                if not claims:
                    return ctx
                evidence_by_claim = ctx.get_extra("evidence_by_claim")
                if hasattr(self.agent, "enrich_claims_post_evidence"):
                    enriched = await self.agent.enrich_claims_post_evidence(
                        claims,
                        lang=ctx.lang,
                        evidence_by_claim=evidence_by_claim,
                    )
                    Trace.event(
                        "claim_enrichment.completed",
                        {"claims": len(enriched or [])},
                    )
                    return ctx.with_update(claims=enriched)
                return ctx

            def build_claims_contract(
                claims_list: list[dict[str, Any]],
                anchor_id: str | None,
            ) -> Claims:
                items: list[ClaimItem] = []
                for idx, claim in enumerate(claims_list):
                    if not isinstance(claim, dict):
                        continue
                    cid = str(claim.get("id") or claim.get("claim_id") or f"c{idx + 1}")
                    text = (claim.get("normalized_text") or claim.get("text") or "").strip()
                    span_start = claim.get("span_start")
                    span_end = claim.get("span_end")
                    lang = claim.get("lang") or claim.get("language") or ctx.lang
                    items.append(
                        ClaimItem(
                            id=cid,
                            text=text,
                            span_start=int(span_start) if isinstance(span_start, int) else None,
                            span_end=int(span_end) if isinstance(span_end, int) else None,
                            lang=str(lang) if lang else None,
                        )
                    )
                return Claims(claims=tuple(items), anchor_claim_id=anchor_id)

            # Check if claims are preloaded (e.g., from deep mode first pass)
            preloaded_claims = ctx.get_extra("preloaded_claims")
            if preloaded_claims and isinstance(preloaded_claims, list) and len(preloaded_claims) > 0:
                # Use preloaded claims, skip extraction
                claims = preloaded_claims
                Trace.event("extract_claims.using_preloaded", {
                    "count": len(claims),
                    "claim_ids": [c.get("id") for c in claims[:10]],
                })
                # Still need to select anchor claim
                anchor_claim = max(claims, key=lambda c: float(c.get("importance", 0.5)))
                anchor_claim_id = str(anchor_claim.get("id", "c1"))
                
                Trace.event(
                    "extract_claims.completed",
                    {
                        "total_claims": len(claims),
                        "eligible_claims": len(claims),
                        "anchor_claim_id": anchor_claim_id,
                        "preloaded": True,
                    },
                )

                claims_contract = build_claims_contract(claims, anchor_claim_id)

                return (
                    ctx.with_update(claims=claims)
                    .set_extra("anchor_claim_id", anchor_claim_id)
                    .set_extra("eligible_claims", claims)
                    .set_extra("_extracted_claims", claims)
                    .set_extra(CLAIMS_KEY, claims_contract)
                )
            
            fact = ctx.get_extra("prepared_fact") or ctx.get_extra("raw_fact", "")
            progress_callback = ctx.get_extra("progress_callback")

            if progress_callback:
                await progress_callback("extracting_claims")

            # Delegate to agent wrapper
            result_tuple = await self.agent.extract_claims(
                text=fact,
                lang=ctx.lang,
            )
            claims, check_oracle, intent, fast_query = result_tuple
            if not claims:
                # Fallback: use full text as single claim
                claims = [{"id": "c1", "text": fact[:500], "importance": 1.0}]

            # Semantic claim dedup right after extraction (pre-oracle/graph/search).
            # This reduces cost + prevents anchor/secondary duplicates.
            from spectrue_core.verification.claims.claim_dedup import dedup_claims_post_extraction_async
            try:
                before_n = len(claims)
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
                                for p in dedup_pairs[:50]  # trace safety cap
                            ],
                        },
                    )
                else:
                    Trace.event(
                        "claims.dedup_post_extraction",
                        {"before": before_n, "after": after_n, "removed": 0, "tau": 0.90, "pairs": []},
                    )
            except Exception as e:
                # Non-fatal: if embeddings unavailable or error occurs, proceed with raw claims.
                Trace.event("claims.dedup_post_extraction.failed", {"error": str(e)})

            # Select anchor claim (highest importance)
            anchor_claim = max(claims, key=lambda c: float(c.get("importance", 0.5)))
            anchor_claim_id = str(anchor_claim.get("id", "c1"))

            # Filter eligible claims (importance >= 0.3)
            eligible_claims = [
                c for c in claims
                if float(c.get("importance", 0.5)) >= 0.3
            ]

            Trace.event(
                "extract_claims.completed",
                {
                    "total_claims": len(claims),
                    "eligible_claims": len(eligible_claims),
                    "anchor_claim_id": anchor_claim_id,
                },
            )

            claims_contract = build_claims_contract(claims, anchor_claim_id)

            return (
                ctx.with_update(claims=claims)
                .set_extra("anchor_claim_id", anchor_claim_id)
                .set_extra("eligible_claims", eligible_claims)
                .set_extra("check_oracle", check_oracle)
                .set_extra("article_intent", str(intent))
                .set_extra("fast_query", fast_query)
                .set_extra("search_queries", [fast_query] if fast_query else [fact])
                .set_extra(CLAIMS_KEY, claims_contract)
            )

        except Exception as e:
            logger.exception("[ExtractClaimsStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
