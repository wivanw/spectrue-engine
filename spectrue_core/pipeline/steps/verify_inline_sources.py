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
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class VerifyInlineSourcesStep:
    """
    Verify inline sources against extracted claims.

    Filters inline sources for relevance to claims.
    Updates final_sources with verified inline sources.

    Context Input:
        - inline_sources (candidates)
        - claims
        - prepared_fact (as fact)

    Context Output:
        - sources (appended with verified inline sources)
    """

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    config: Any  # SpectrueConfig
    name: str = "verify_inline_sources"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Verify inline sources."""
        from spectrue_core.verification.pipeline_input import verify_inline_sources
        from spectrue_core.runtime_config import ContentBudgetConfig

        try:
            inline_sources = ctx.get_extra("inline_sources", [])
            claims = ctx.claims
            fact = ctx.get_extra("prepared_fact", "")
            current_sources = ctx.sources or []
            progress_callback = ctx.get_extra("progress_callback")

            if not inline_sources:
                return ctx

            # Extract config
            content_budget_cfg = getattr(getattr(self.config, "runtime", None), "content_budget", None)
            if not isinstance(content_budget_cfg, ContentBudgetConfig):
                content_budget_cfg = ContentBudgetConfig()

            # Verify (using helper that handles logic)
            # Note: verified_inline_sources are NEW sources to add
            verified = await verify_inline_sources(
                inline_sources=inline_sources,
                claims=claims,
                fact=fact,
                agent=self.agent,
                search_mgr=self.search_mgr,
                content_budget_config=content_budget_cfg,
                progress_callback=progress_callback,
            )

            # Fallback logic if verification returned nothing but we had candidates and NO claims?
            if not verified and inline_sources and not claims:
                 for src in inline_sources:
                    src["is_primary"] = False
                    src["is_relevant"] = True
                    src["is_trusted"] = False
                    src["source_type"] = "inline"
                 verified = inline_sources
            
            new_sources = current_sources + verified
            
            Trace.event(
                "verify_inline_sources.step_completed",
                {
                    "candidates": len(inline_sources),
                    "verified": len(verified),
                    "total_sources": len(new_sources),
                    "fallback_triggered": (not claims and bool(inline_sources)),
                },
            )

            return ctx.with_update(sources=new_sources)

        except Exception as e:
            logger.warning("[VerifyInlineSourcesStep] Non-fatal failure: %s", e)
            return ctx
