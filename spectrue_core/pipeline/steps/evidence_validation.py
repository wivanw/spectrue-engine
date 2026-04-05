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
Evidence Validation Step

Filters out invalid or irrelevant evidence before it reaches the judge.
Removes:
- Evidence with IRRELEVANT stance
- Evidence with no usable content (no quote, snippet, or content)
"""

from __future__ import annotations

from dataclasses import dataclass

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.utils.trace import Trace


@dataclass
class EvidenceValidationStep(Step):
    """Filter out invalid/irrelevant evidence before judging."""

    name: str = "evidence_validation"
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        sources = ctx.sources or []
        if not sources:
            return ctx

        valid: list[dict] = []
        rejected = {
            "irrelevant_stance": 0,
            "empty_content": 0,
        }

        for s in sources:
            if not isinstance(s, dict):
                continue

            # Reject IRRELEVANT stance
            stance = str(s.get("stance") or "").upper()
            if stance == "IRRELEVANT":
                rejected["irrelevant_stance"] += 1
                continue

            # Must have some usable content
            has_content = bool(
                s.get("quote")
                or s.get("quote_span")
                or s.get("snippet")
                or s.get("content")
                or s.get("excerpt")
            )
            if not has_content:
                rejected["empty_content"] += 1
                continue

            valid.append(s)

        total_rejected = sum(rejected.values())
        Trace.event(
            "evidence_validation.completed",
            {
                "input": len(sources),
                "kept": len(valid),
                "rejected_total": total_rejected,
                **rejected,
            },
        )

        if total_rejected > 0:
            ctx = ctx.with_update(sources=valid)

            # Update evidence_by_claim if present
            by_claim = ctx.get_extra("evidence_by_claim")
            if isinstance(by_claim, dict):
                valid_urls = {s.get("url") for s in valid if s.get("url")}
                updated = {}
                for cid, items in by_claim.items():
                    if isinstance(items, list):
                        updated[cid] = [
                            i for i in items
                            if not i.get("url") or i.get("url") in valid_urls
                        ]
                    else:
                        updated[cid] = items
                ctx = ctx.set_extra("evidence_by_claim", updated)

        return ctx
