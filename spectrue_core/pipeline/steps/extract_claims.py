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
    name: str = "extract_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Extract claims from fact."""
        try:
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

            return (
                ctx.with_update(claims=claims)
                .set_extra("anchor_claim_id", anchor_claim_id)
                .set_extra("eligible_claims", eligible_claims)
                .set_extra("check_oracle", check_oracle)
                .set_extra("article_intent", str(intent))
                .set_extra("fast_query", fast_query)
                .set_extra("search_queries", [fast_query] if fast_query else [fact])
            )

        except Exception as e:
            logger.exception("[ExtractClaimsStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
